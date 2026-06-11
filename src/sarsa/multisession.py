# Author: Yuan Zhao <yuan.zhao@nih.gov>
# Affiliation: Machine Learning Core, NIMH
"""
Multi-session pooled SARSA fitting for a single subject.

This module fits several sessions of one subject jointly so that a chosen
subset of parameters (``alpha``, ``beta``, ``gamma``, and/or user-defined
parameters) is **shared** across sessions (one point estimate), while the
remaining parameters stay **session-specific** (one estimate per session).

Design summary (see ``spec/02_multisession_pooling.md``):

- A subject is an ordered ``list[Session]`` where ``Session = list[Quintuple]``.
- A length-``P`` boolean ``share_mask`` (``P = 3 + n_user_params``) marks each
  base parameter shared (``True``) or session-specific (``False``).
- The optimizer vector is ``[shared] + [per-session specific] * n_sessions``.
- ``Q`` is threaded across sessions in order; at each gap a fixed rule is
  applied (``carry`` / ``decay`` / ``reset``). No quintuple straddles a gap, so
  no TD update happens across the boundary.
- ``gap_decay`` is always a fixed constant in ``[0, 1]`` and is never estimated.
- The objective is one pooled cross-entropy over all stacked trials.

The vanilla SARSA kernel (:func:`sarsa.sarsa.run` / :func:`sarsa.sarsa.update`)
is reused unchanged per session, as is the parameter-subspace machinery
(:func:`materialize_params`, :func:`select_trainable_params`, etc.).
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from .sarsa import (
    DEFAULT_POLICY_BETA,
    SARSA_PARAM_BOUNDS,
    Quintuple,
    _validate_quintuples,
    cross_entropy,
    materialize_params,
    merge,
    resolve_static_params,
    run,
    select_trainable_bounds,
    select_trainable_params,
    to_prob,
    validate_fixed_params_against_bounds,
    warn_if_sarsa_params_hit_bounds,
)

logger = logging.getLogger(__name__)

# Type aliases (internal documentation only; users pass plain nested lists).
Session = list[Quintuple]
Subject = list[Session]

GAP_RULES = ("carry", "decay", "reset")
_N_CANONICAL = len(SARSA_PARAM_BOUNDS)


@dataclass
class SubjectFitResult:
    """Result of a pooled multi-session fit.

    Attributes
    ----------
    shared_params : NDArray
        Fitted values for the shared base parameters, in base order
        (the subset of ``[alpha, beta, gamma, *user_params]`` where
        ``share_mask`` is ``True``).
    session_params : list[NDArray]
        Per-session full base parameter vectors, each of length ``P``.
    gap_decay : float
        The fixed gap-decay constant used (``1.0`` unless overridden).
    loss : float
        Final pooled cross-entropy loss.
    q_trajectories : list[NDArray]
        Per-session Q trajectories, each of shape ``(T_s + 1, *state_dims, A)``.
    action_probs : list[NDArray]
        Per-session action probabilities, each of shape ``(T_s, A)``.
    """

    shared_params: NDArray
    session_params: list[NDArray]
    gap_decay: float
    loss: float
    q_trajectories: list[NDArray]
    action_probs: list[NDArray]


# ---------------------------------------------------------------------------
# Parameter layout helpers
# ---------------------------------------------------------------------------


def _shared_indices(share_mask: Sequence[bool]) -> list[int]:
    return [i for i, m in enumerate(share_mask) if m]


def _spec_indices(share_mask: Sequence[bool]) -> list[int]:
    return [i for i, m in enumerate(share_mask) if not m]


def optimizer_vector_length(share_mask: Sequence[bool], n_sessions: int) -> int:
    """Return the length of the full optimizer vector for the given layout."""
    n_shared = sum(bool(m) for m in share_mask)
    n_spec = len(share_mask) - n_shared
    return n_shared + n_sessions * n_spec


def scatter_session_params(
    full_vec: Sequence[float] | NDArray,
    share_mask: Sequence[bool],
    n_sessions: int,
) -> list[NDArray]:
    """Expand the flat optimizer vector into per-session base vectors.

    Parameters
    ----------
    full_vec : Sequence[float] or NDArray
        Optimizer vector laid out as
        ``[shared] + [session 0 specific] + ... + [session S-1 specific]``.
    share_mask : Sequence[bool]
        Length-``P`` mask; ``True`` marks a shared base parameter.
    n_sessions : int
        Number of sessions ``S``.

    Returns
    -------
    list[NDArray]
        ``[theta_0, ..., theta_{S-1}]``, each a length-``P`` base vector.
    """
    full_vec = np.asarray(full_vec, dtype=float)
    P = len(share_mask)
    shared_idx = _shared_indices(share_mask)
    spec_idx = _spec_indices(share_mask)
    n_shared = len(shared_idx)
    n_spec = len(spec_idx)

    expected = n_shared + n_sessions * n_spec
    if full_vec.ndim != 1 or len(full_vec) != expected:
        raise ValueError(
            "full_vec length does not match layout; "
            f"got {len(full_vec)} and expected {expected}"
        )

    shared_vals = full_vec[:n_shared]
    thetas: list[NDArray] = []
    for s in range(n_sessions):
        start = n_shared + s * n_spec
        spec_vals = full_vec[start : start + n_spec]
        theta = np.empty(P, dtype=float)
        if n_shared:
            theta[shared_idx] = shared_vals
        if n_spec:
            theta[spec_idx] = spec_vals
        thetas.append(theta)
    return thetas


def gather_session_params(
    thetas: Sequence[Sequence[float] | NDArray],
    share_mask: Sequence[bool],
) -> NDArray:
    """Pack per-session base vectors into the flat optimizer vector.

    Shared values are taken from the first session; session-specific values are
    taken from each session in order. Inverse of :func:`scatter_session_params`.
    """
    shared_idx = _shared_indices(share_mask)
    spec_idx = _spec_indices(share_mask)
    thetas = [np.asarray(t, dtype=float) for t in thetas]
    parts: list[NDArray] = [thetas[0][shared_idx]] if shared_idx else [np.empty(0)]
    for theta in thetas:
        if spec_idx:
            parts.append(theta[spec_idx])
    return np.concatenate(parts) if parts else np.empty(0)


def _gap_transform(
    q_end: NDArray, q0: NDArray, gap_rule: str, gap_decay: float
) -> NDArray:
    """Transform Q at a session boundary according to ``gap_rule``."""
    if gap_rule == "carry":
        return q_end
    if gap_rule == "decay":
        return gap_decay * q_end
    if gap_rule == "reset":
        return q0.copy()
    raise ValueError(f"gap_rule must be one of {GAP_RULES}; got {gap_rule!r}")


# ---------------------------------------------------------------------------
# Forward pass and loss
# ---------------------------------------------------------------------------


def run_subject(
    full_vec: Sequence[float] | NDArray,
    sessions: Sequence[Sequence[Quintuple]],
    q0: NDArray,
    share_mask: Sequence[bool],
    *,
    gap_rule: str = "carry",
    gap_decay: float = 1.0,
    transition_reward_func: Callable | None = None,
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """Run SARSA across all sessions, threading Q across gaps.

    Parameters
    ----------
    full_vec : Sequence[float] or NDArray
        Full optimizer vector (see :func:`scatter_session_params`).
    sessions : Sequence[Sequence[Quintuple]]
        Ordered sessions for one subject.
    q0 : NDArray
        Initial Q-function for the first session.
    share_mask : Sequence[bool]
        Length-``P`` shared/session-specific mask.
    gap_rule : str, optional
        ``"carry"`` (default), ``"decay"``, or ``"reset"``.
    gap_decay : float, optional
        Fixed decay constant used when ``gap_rule="decay"``.
    transition_reward_func : Callable or None, optional
        Optional reward callback ``(user_params, s1, a1, s2) -> (s2, reward)``.

    Returns
    -------
    tuple[list[NDArray], list[NDArray], list[NDArray]]
        Per-session ``(q_trajectories, log_probs, td_errors)``.
    """
    thetas = scatter_session_params(full_vec, share_mask, len(sessions))
    q = q0
    qs_list: list[NDArray] = []
    logprob_list: list[NDArray] = []
    error_list: list[NDArray] = []
    for theta, session in zip(thetas, sessions):
        qs, logprob, error = run(theta, session, q, transition_reward_func)
        qs_list.append(qs)
        logprob_list.append(logprob)
        error_list.append(error)
        q = _gap_transform(qs[-1], q0, gap_rule, gap_decay)
    return qs_list, logprob_list, error_list


def run_and_loss_subject(
    full_vec: Sequence[float] | NDArray,
    sessions: Sequence[Sequence[Quintuple]],
    q0: NDArray,
    share_mask: Sequence[bool],
    *,
    gap_rule: str = "carry",
    gap_decay: float = 1.0,
    transition_reward_func: Callable | None = None,
) -> float:
    """Run all sessions and return the pooled cross-entropy loss."""
    _, logprob_list, _ = run_subject(
        full_vec,
        sessions,
        q0,
        share_mask,
        gap_rule=gap_rule,
        gap_decay=gap_decay,
        transition_reward_func=transition_reward_func,
    )
    actions = np.array([q.a1 for session in sessions for q in session], dtype=np.int_)
    logprob = np.concatenate(logprob_list, axis=0)
    return float(cross_entropy(logprob, actions))


def _run_and_loss_subject_trainable(
    trainable_params: NDArray,
    static_params: Sequence[float | None],
    sessions: Sequence[Sequence[Quintuple]],
    q0: NDArray,
    share_mask: Sequence[bool],
    gap_rule: str,
    gap_decay: float,
    transition_reward_func: Callable | None,
) -> float:
    """Evaluate the pooled loss in the reduced trainable subspace."""
    full_vec = materialize_params(trainable_params, static_params)
    return run_and_loss_subject(
        full_vec,
        sessions,
        q0,
        share_mask,
        gap_rule=gap_rule,
        gap_decay=gap_decay,
        transition_reward_func=transition_reward_func,
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _expand_static(
    base_static: Sequence[float | None],
    share_mask: Sequence[bool],
    n_sessions: int,
) -> list[float | None]:
    """Expand base-level static values to the optimizer-vector layout."""
    shared_idx = _shared_indices(share_mask)
    spec_idx = _spec_indices(share_mask)
    expanded: list[float | None] = [base_static[i] for i in shared_idx]
    for _ in range(n_sessions):
        expanded.extend(base_static[i] for i in spec_idx)
    return expanded


def _expand_bounds(
    base_bounds: Sequence[tuple[float | None, float | None]],
    share_mask: Sequence[bool],
    n_sessions: int,
) -> list[tuple[float | None, float | None]]:
    """Expand base-level bounds to the optimizer-vector layout."""
    shared_idx = _shared_indices(share_mask)
    spec_idx = _spec_indices(share_mask)
    expanded: list[tuple[float | None, float | None]] = [
        base_bounds[i] for i in shared_idx
    ]
    for _ in range(n_sessions):
        expanded.extend(base_bounds[i] for i in spec_idx)
    return expanded


def _resolve_share_mask(
    share_mask: Sequence[bool] | None, param_count: int
) -> list[bool]:
    if share_mask is None:
        return [True] * param_count
    share_mask = [bool(m) for m in share_mask]
    if len(share_mask) != param_count:
        raise ValueError(
            "share_mask must match base parameter count; "
            f"got {len(share_mask)} and expected {param_count}"
        )
    return share_mask


def _build_session_thetas(
    p0: NDArray, param_count: int, n_sessions: int
) -> list[NDArray]:
    """Build per-session base p0 vectors from a base vector or per-session list."""
    p0 = np.asarray(p0, dtype=float)
    if p0.ndim == 1:
        if len(p0) != param_count:
            raise ValueError(
                "p0 length must match base parameter count; "
                f"got {len(p0)} and expected {param_count}"
            )
        return [p0.copy() for _ in range(n_sessions)]
    if p0.ndim == 2:
        if p0.shape != (n_sessions, param_count):
            raise ValueError(
                "per-session p0 must have shape (n_sessions, param_count); "
                f"got {p0.shape} and expected {(n_sessions, param_count)}"
            )
        return [row.copy() for row in p0]
    raise ValueError("p0 must be 1-D (broadcast) or 2-D (per-session)")


def fit_subject(
    sessions: Sequence[Sequence[Quintuple]],
    q0: NDArray,
    p0: NDArray,
    share_mask: Sequence[bool] | None = None,
    static_params: Sequence[float | None] | None = None,
    transition_reward_func: Callable | None = None,
    user_param_bounds: Sequence[tuple[float | None, float | None]] = (),
    *,
    fit_beta: bool = False,
    policy_beta: float = DEFAULT_POLICY_BETA,
    gap_rule: str = "carry",
    gap_decay: float = 1.0,
) -> SubjectFitResult:
    """Fit pooled SARSA parameters across one subject's sessions.

    Parameters
    ----------
    sessions : Sequence[Sequence[Quintuple]]
        Ordered sessions (earliest first); each session is a list of quintuples.
    q0 : NDArray
        Initial Q-function for the first session.
    p0 : NDArray
        Initial guess. Either a length-``P`` base vector (broadcast to every
        session) or a ``(n_sessions, P)`` array of per-session base vectors,
        where ``P = 3 + len(user_param_bounds)``.
    share_mask : Sequence[bool] or None, optional
        Length-``P`` mask; ``True`` shares the parameter across sessions.
        Defaults to all-shared.
    static_params : Sequence[float | None] or None, optional
        Base-level fixed parameter values (length ``P``). A fixed shared
        parameter fixes its single value; a fixed session-specific parameter is
        broadcast to all sessions.
    transition_reward_func : Callable or None, optional
        Reward callback ``(user_params, s1, a1, s2) -> (s2, reward)``. Required
        when user parameters are present.
    user_param_bounds : Sequence[tuple[float | None, float | None]], optional
        Bounds for user-defined parameters appended after the canonical block.
    fit_beta : bool, optional
        If ``False`` (default), beta is treated as a fixed policy hyperparameter
        at the base level. If ``True``, beta is trainable unless fixed in
        ``static_params``.
    policy_beta : float, optional
        Fixed beta value used when ``fit_beta=False``.
    gap_rule : str, optional
        Q transform at session gaps: ``"carry"`` (default), ``"decay"``, or
        ``"reset"``.
    gap_decay : float, optional
        Fixed decay constant in ``[0, 1]`` used when ``gap_rule="decay"``.
        Never estimated.

    Returns
    -------
    SubjectFitResult
        Fitted shared params, per-session params, loss, and per-session
        trajectories / action probabilities.

    Raises
    ------
    ValueError
        For empty input, mask/length mismatches, invalid ``gap_rule`` /
        ``gap_decay``, bound violations, or user parameters without a reward
        callback.
    """
    sessions = list(sessions)
    if len(sessions) == 0:
        raise ValueError("sessions must be non-empty")
    n_sessions = len(sessions)

    if gap_rule not in GAP_RULES:
        raise ValueError(f"gap_rule must be one of {GAP_RULES}; got {gap_rule!r}")
    if not np.isfinite(gap_decay) or not (0.0 <= gap_decay <= 1.0):
        raise ValueError(f"gap_decay must lie in [0, 1]; got {gap_decay}")

    user_param_bounds = tuple(user_param_bounds)
    param_count = _N_CANONICAL + len(user_param_bounds)
    if transition_reward_func is None and param_count != _N_CANONICAL:
        raise ValueError("user-defined parameters require transition_reward_func")

    share_mask = _resolve_share_mask(share_mask, param_count)

    # Validate every session against the shared Q-table geometry.
    for session in sessions:
        _validate_quintuples(session, q0)

    # Resolve base-level static (handles fit_beta / policy_beta) and bounds.
    base_static = resolve_static_params(
        static_params,
        param_count,
        fit_beta=fit_beta,
        policy_beta=policy_beta,
    )
    base_bounds = list(SARSA_PARAM_BOUNDS) + list(user_param_bounds)
    validate_fixed_params_against_bounds(base_static, base_bounds)

    # Build full optimizer vector from per-session base p0, then expand layout.
    thetas0 = _build_session_thetas(p0, param_count, n_sessions)
    full_p0 = gather_session_params(thetas0, share_mask)
    expanded_static = _expand_static(base_static, share_mask, n_sessions)
    expanded_bounds = _expand_bounds(base_bounds, share_mask, n_sessions)

    full_p0 = merge(full_p0, expanded_static)
    trainable_p0 = select_trainable_params(full_p0, expanded_static)
    trainable_bounds = select_trainable_bounds(expanded_bounds, expanded_static)

    if len(trainable_p0) == 0:
        full_params = full_p0
        loss = run_and_loss_subject(
            full_params,
            sessions,
            q0,
            share_mask,
            gap_rule=gap_rule,
            gap_decay=gap_decay,
            transition_reward_func=transition_reward_func,
        )
        logger.info("Optimizer skipped: all parameters fixed")
    else:
        res = optimize.minimize(
            _run_and_loss_subject_trainable,
            x0=trainable_p0,
            args=(
                expanded_static,
                sessions,
                q0,
                share_mask,
                gap_rule,
                gap_decay,
                transition_reward_func,
            ),
            bounds=trainable_bounds,
            method="L-BFGS-B",
        )
        loss = float(res.fun)
        full_params = materialize_params(res.x, expanded_static)
        logger.info(
            "Optimizer finished: success=%s, message=%s", res.success, res.message
        )

    session_params = scatter_session_params(full_params, share_mask, n_sessions)

    # Boundary diagnostics: check shared canonical params once (session 0) and
    # session-specific canonical params per session, to avoid duplicate warnings.
    for s, theta in enumerate(session_params):
        static_canon: list[float | None] = []
        for i in range(_N_CANONICAL):
            if share_mask[i] and s > 0:
                # Shared canonical already checked on session 0; skip here.
                static_canon.append(float(theta[i]))
            else:
                static_canon.append(base_static[i])
        warn_if_sarsa_params_hit_bounds(theta[:_N_CANONICAL], static_canon)

    qs_list, logprob_list, _ = run_subject(
        full_params,
        sessions,
        q0,
        share_mask,
        gap_rule=gap_rule,
        gap_decay=gap_decay,
        transition_reward_func=transition_reward_func,
    )
    action_probs = [to_prob(lp) for lp in logprob_list]
    shared_params = full_params[: sum(share_mask)]

    return SubjectFitResult(
        shared_params=shared_params,
        session_params=session_params,
        gap_decay=gap_decay,
        loss=loss,
        q_trajectories=qs_list,
        action_probs=action_probs,
    )
