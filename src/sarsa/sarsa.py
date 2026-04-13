# Author: Yuan Zhao <yuan.zhao@nih.gov>
# Affiliation: Machine Learning Core, NIMH
"""
SARSA

This implementation is designed to be independent from the interpretation of state and action.
It only requires the state and action to be integer NumPy arrays.

The flat parameter vector is the concatenation of two blocks:

- ``sarsa_params`` -- parameters interpreted by the vanilla SARSA kernel
- ``user_params`` -- user-defined extension parameters consumed by callbacks

The canonical SARSA parameter order is:

- ``sarsa_params[ParamIndex.alpha]`` -- **alpha** (learning rate)
- ``sarsa_params[ParamIndex.beta]`` -- **beta** (inverse temperature for the softmax policy)
- ``sarsa_params[ParamIndex.gamma]`` -- **gamma** (discount / decay factor)

This split isolates user-defined extension state from the vanilla kernel; it
does not change the canonical ``alpha``, ``beta``, ``gamma`` semantics of SARSA
itself.

By default, :func:`fit` treats ``beta`` as a fixed policy hyperparameter and
optimizes the remaining free coordinates only. Use ``fit_beta=True`` to opt in
to fitting ``beta`` explicitly.

Use :func:`concat_params` and :func:`split_params` to construct and unpack the
flat optimizer vector.
"""

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import log_softmax

logger = logging.getLogger(__name__)

EPS = 1e-8  # Minimum positive value
DEFAULT_POLICY_BETA = 5.0  # Large but finite beta for near-greedy default policy
SARSA_PARAM_BOUNDS = [
    (EPS, None),
    (EPS, None),
    (EPS, 1 - EPS),
]  # Bounds for vanilla SARSA parameters
PARAM_BOUNDS = SARSA_PARAM_BOUNDS  # Backward-compatible alias


class ParamIndex(IntEnum):
    """SARSA parameter indices.

    Attributes
    ----------
    alpha : int
        Learning rate index.
    beta : int
        Inverse temperature index.
    gamma : int
        Discount/decay index.
    """

    alpha = 0  # learning rate
    beta = 1  # inverse temperature
    gamma = 2  # decay


@dataclass
class Quintuple:
    """Container describing a single SARSA transition."""

    s1: NDArray
    a1: int
    r2: float
    s2: NDArray
    a2: int


def concat_params(
    sarsa_params: Sequence[float] | NDArray,
    user_params: Sequence[float] | NDArray = (),
) -> NDArray:
    """Concatenate SARSA-owned and user-defined parameter blocks.

    Parameters
    ----------
    sarsa_params : Sequence[float] or NDArray
        Flat block of parameters consumed by the vanilla SARSA kernel.
    user_params : Sequence[float] or NDArray, optional
        Flat block of user-defined extension parameters appended after
        ``sarsa_params`` in the optimizer vector.

    Returns
    -------
    NDArray
        Concatenated flat parameter vector.

    Raises
    ------
    ValueError
        If either block is not one-dimensional, or if ``sarsa_params`` does not
        match the canonical SARSA parameter count.
    """
    sarsa_params = np.asarray(sarsa_params, dtype=float)
    user_params = np.asarray(user_params, dtype=float)
    if sarsa_params.ndim != 1 or user_params.ndim != 1:
        raise ValueError("parameter blocks must be one-dimensional")
    if len(sarsa_params) != len(SARSA_PARAM_BOUNDS):
        raise ValueError(
            "sarsa_params must match canonical SARSA parameter count; "
            f"got {len(sarsa_params)} and expected {len(SARSA_PARAM_BOUNDS)}"
        )
    return np.concatenate((sarsa_params, user_params))


def split_params(params: Sequence[float] | NDArray) -> tuple[NDArray, NDArray]:
    """Split a flat optimizer vector into SARSA-owned and user-defined blocks.

    Parameters
    ----------
    params : Sequence[float] or NDArray
        Flat parameter vector containing the SARSA block followed by any
        user-defined extension parameters.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(sarsa_params, user_params)`` as one-dimensional arrays.

    Raises
    ------
    ValueError
        If ``params`` is not one-dimensional or shorter than the canonical SARSA
        parameter count.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 1:
        raise ValueError("params must be one-dimensional")
    sarsa_param_count = len(SARSA_PARAM_BOUNDS)
    if len(params) < sarsa_param_count:
        raise ValueError(
            "params must include the full SARSA parameter block; "
            f"got {len(params)} and expected at least {sarsa_param_count}"
        )
    return params[:sarsa_param_count].copy(), params[sarsa_param_count:].copy()


def action_logprob(sarsa_params: NDArray, v: NDArray) -> NDArray:
    """Compute softmax log-probabilities for each action.

    Parameters
    ----------
    sarsa_params : NDArray
        SARSA-owned parameter block with the inverse temperature stored at
        ``ParamIndex.beta``.
    v : NDArray
        Action-value estimates prior to scaling.

    Returns
    -------
    NDArray
        Log-probabilities over the action set after softmaxing ``v`` by ``beta``.
    """
    beta = sarsa_params[ParamIndex.beta]
    return log_softmax(v * beta)


def to_prob(p: NDArray) -> NDArray:
    """Convert log-probabilities into probabilities.

    Parameters
    ----------
    p : NDArray
        Log-probabilities over the action set.

    Returns
    -------
    NDArray
        Probability distribution matching ``p``.
    """
    return np.exp(p)


def cross_entropy(inputs: NDArray, targets: NDArray) -> np.floating:
    """Compute cross-entropy loss against observed actions.

    Parameters
    ----------
    inputs : NDArray
        Log-probabilities predicted for each action.
    targets : NDArray
        Indices of the actions actually taken.

    Returns
    -------
    float
        Mean negative log-likelihood of the target actions.
    """
    ce = np.take_along_axis(inputs, np.expand_dims(targets, axis=1), axis=1)
    return -np.nanmean(ce)


def merge(params: NDArray, static: Sequence[float | None]) -> NDArray:
    """Combine full-length parameters with optional fixed values.

    Parameters
    ----------
    params : NDArray
        Full parameter vector whose entries may be overwritten by ``static``.
    static : Sequence[float | None]
        Fixed values for each parameter position; ``None`` keeps the corresponding
        value from ``params``.

    Returns
    -------
    NDArray
        Parameter vector with static overrides applied.

    Raises
    ------
    ValueError
        If ``params`` and ``static`` lengths differ.
    """
    if len(params) != len(static):
        raise ValueError(
            "params and static must have the same length; "
            f"got {len(params)} and {len(static)}"
        )
    return np.array(
        [p if s is None else s for p, s in zip(params, static)], dtype=float
    )


def select_trainable_params(
    full_params: Sequence[float] | NDArray,
    static: Sequence[float | None],
) -> NDArray:
    """Extract the trainable coordinates from a full parameter vector."""
    full_params = np.asarray(full_params, dtype=float)
    if full_params.ndim != 1:
        raise ValueError("full_params must be one-dimensional")
    if len(full_params) != len(static):
        raise ValueError(
            "full_params and static must have the same length; "
            f"got {len(full_params)} and {len(static)}"
        )
    return np.array(
        [p for p, s in zip(full_params, static) if s is None], dtype=float
    )


def materialize_params(
    trainable_params: Sequence[float] | NDArray,
    static: Sequence[float | None],
) -> NDArray:
    """Reconstruct the full parameter vector from trainable coordinates."""
    trainable_params = np.asarray(trainable_params, dtype=float)
    if trainable_params.ndim != 1:
        raise ValueError("trainable_params must be one-dimensional")

    full = np.empty(len(static), dtype=float)
    j = 0
    for i, s in enumerate(static):
        if s is None:
            if j >= len(trainable_params):
                raise ValueError(
                    "trainable_params length does not match free parameter slots"
                )
            full[i] = trainable_params[j]
            j += 1
        else:
            full[i] = float(s)
    if j != len(trainable_params):
        raise ValueError(
            "trainable_params length does not match free parameter slots"
        )
    return full


def select_trainable_bounds(
    full_bounds: Sequence[tuple[float | None, float | None]],
    static: Sequence[float | None],
) -> list[tuple[float | None, float | None]]:
    """Extract bounds for trainable coordinates only."""
    if len(full_bounds) != len(static):
        raise ValueError(
            "full_bounds and static must have the same length; "
            f"got {len(full_bounds)} and {len(static)}"
        )
    return [bound for bound, s in zip(full_bounds, static) if s is None]


def resolve_static_params(
    static_params: Sequence[float | None] | None,
    param_count: int,
    *,
    fit_beta: bool,
    policy_beta: float,
) -> list[float | None]:
    """Resolve fixed parameters, including the default beta hyperparameter."""
    if static_params is None:
        resolved = [None] * param_count
    else:
        resolved = list(static_params)
    if len(resolved) != param_count:
        raise ValueError(
            "static_params must match p0 length; "
            f"got {len(resolved)} and {param_count}"
        )
    if not np.isfinite(policy_beta) or policy_beta <= EPS:
        raise ValueError(
            f"policy_beta must be finite and greater than {EPS}; got {policy_beta}"
        )

    beta_index = ParamIndex.beta
    if resolved[beta_index] is not None:
        if fit_beta:
            warnings.warn(
                "fit_beta=True ignored because static_params fixes beta.",
                UserWarning,
                stacklevel=2,
            )
    elif not fit_beta:
        resolved[beta_index] = float(policy_beta)
    return resolved


def update(sarsa_params: NDArray, quintuple: Quintuple, q: NDArray) -> tuple[NDArray, float]:
    """Apply the SARSA update for a single transition.

    Parameters
    ----------
    sarsa_params : NDArray
        SARSA-owned parameter block containing the learning rate and discount factor.
    quintuple : Quintuple
        Transition describing state-action pairs and the next state.
    q : NDArray
        Q-function prior to applying the update.

    Returns
    -------
    NDArray
        Updated Q-function after the SARSA step.
    float
        Temporal-difference error produced by the update.
    """
    # consequent reward transitioning from s1 to s2
    alpha = sarsa_params[ParamIndex.alpha]
    gamma = sarsa_params[ParamIndex.gamma]
    q_new = q.copy()
    s1 = quintuple.s1
    a1 = quintuple.a1
    s2 = quintuple.s2
    a2 = quintuple.a2
    r = quintuple.r2

    error = r + gamma * q[*s2, a2] - q[*s1, a1]  # TD error

    q_new[*s1, a1] = q[*s1, a1] + alpha * error  # Update
    return q_new, error


def _validate_quintuples(quintuples: Sequence[Quintuple], q0: NDArray) -> None:
    """Validate quintuples and Q-table compatibility.

    Parameters
    ----------
    quintuples : Sequence[Quintuple]
        Rollout transitions describing the trajectory to learn from.
    q0 : NDArray
        Initial Q-function prior to any updates.

    Raises
    ------
    ValueError
        If quintuples are empty, state/action indices are not integer arrays,
        or indices fall outside the bounds of ``q0``.
    """
    if len(quintuples) == 0:
        raise ValueError("quintuples must be non-empty")
    sample = quintuples[0]
    if sample.s1.ndim != 1 or sample.s2.ndim != 1:
        raise ValueError("state vectors must be 1-D")
    if not np.issubdtype(sample.s1.dtype, np.integer):
        raise ValueError("state vectors must use integer dtype")
    state_dims = q0.shape[:-1]
    if len(state_dims) != sample.s1.shape[0]:
        raise ValueError(
            "q0 state dimensions must match state vector length; "
            f"got {len(state_dims)} and {sample.s1.shape[0]}"
        )
    action_dim = q0.shape[-1]
    for quintuple in quintuples:
        if not np.issubdtype(quintuple.s1.dtype, np.integer) or not np.issubdtype(
            quintuple.s2.dtype, np.integer
        ):
            raise ValueError("state vectors must use integer dtype")
        if (
            quintuple.s1.shape != sample.s1.shape
            or quintuple.s2.shape != sample.s2.shape
        ):
            raise ValueError("state vectors must be consistent in shape")
        if np.any(quintuple.s1 < 0) or np.any(quintuple.s2 < 0):
            raise ValueError("state indices must be non-negative")
        if np.any(quintuple.s1 >= state_dims) or np.any(quintuple.s2 >= state_dims):
            raise ValueError("state indices out of bounds for q0")
        if not (0 <= quintuple.a1 < action_dim) or not (0 <= quintuple.a2 < action_dim):
            raise ValueError("action indices out of bounds for q0")


def run(
    params: NDArray,
    quintuples: Sequence[Quintuple],
    q0: NDArray,
    transition_reward_func: Callable | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Execute SARSA over a sequence of quintuples.

    Parameters
    ----------
    params : NDArray
        Flat parameter vector passed to the learning rule.
    quintuples : Sequence[Quintuple]
        Rollout transitions describing the trajectory to learn from.
    q0 : NDArray
        Initial Q-function prior to any updates.
    transition_reward_func : Callable or None, optional
        Optional callback with signature
        ``(user_params, s1, a1, s2) -> tuple[NDArray, float]``.
        When omitted, SARSA uses each quintuple's observed ``r2`` directly without
        modifying the input data. When provided, the callback may recompute the reward
        from the user-defined parameter block (and must return the recorded next-state
        ``s2``).

    Returns
    -------
    NDArray
        Trajectory of Q-functions, including the initial state (length ``T + 1``).
    NDArray
        Log-probabilities per timestep for the actions taken.
    NDArray
        Temporal-difference errors per timestep (length ``T``).

    Raises
    ------
    AssertionError
        If ``transition_reward_func`` returns a next-state that differs from
        the quintuple's recorded ``s2``.
    ValueError
        If quintuples are empty, indices are incompatible with ``q0``, vanilla mode
        is asked to consume non-finite observed rewards, or ``params`` cannot be split
        into SARSA and user-defined blocks.
    """
    _validate_quintuples(quintuples, q0)
    sarsa_params, user_params = split_params(params)
    T = len(quintuples)
    qs = np.zeros((T + 1,) + q0.shape)
    error = np.zeros(T)
    q = qs[0] = q0
    logprob = np.zeros((T, q0.shape[-1]))
    for t in range(T):
        quintuple = quintuples[t]
        logprob[t] = action_logprob(sarsa_params, q[*quintuple.s1])
        if transition_reward_func is None:
            if not np.isfinite(quintuple.r2):
                raise ValueError(
                    "vanilla SARSA requires finite observed rewards in quintuple.r2"
                )
            qs[t + 1], error[t] = update(sarsa_params, quintuple, q)
        else:
            s2, r2 = transition_reward_func(
                user_params,
                quintuple.s1,
                quintuple.a1,
                quintuple.s2,
            )  # calculate stepwise net reward on the fly for user-defined extension parameters
            assert np.all(quintuple.s2 == s2)
            # Reward depends on user-defined parameters; recompute each run while
            # avoiding in-place mutation so callers can safely reuse the original
            # quintuples.
            quintuple_with_reward = replace(quintuple, r2=r2)
            qs[t + 1], error[t] = update(sarsa_params, quintuple_with_reward, q)
        q = qs[t + 1]
    return qs, logprob, error


def run_and_loss(
    params: NDArray,
    quintuples: Sequence[Quintuple],
    q0: NDArray,
    transition_reward_func: Callable | None = None,
) -> float:
    """Run SARSA and compute the cross-entropy loss.

    Parameters
    ----------
    params : NDArray
        Full flat parameter vector.
    quintuples : Sequence[Quintuple]
        Rollout transitions describing the trajectory to learn from.
    q0 : NDArray
        Initial Q-function prior to any updates.
    transition_reward_func : Callable or None, optional
        Optional reward recomputation callback operating on ``user_params``.
        When omitted, the loss is computed using observed rewards already stored
        in ``quintuples``.

    Returns
    -------
    float
        Mean cross-entropy loss between predicted and taken actions.
    """
    actions = np.array([q.a1 for q in quintuples], dtype=np.int_)
    _, logprob, _ = run(params, quintuples, q0, transition_reward_func)
    assert len(logprob) == len(actions), f"{len(logprob)}, {len(actions)}"
    ce = cross_entropy(logprob, actions)
    return float(ce)


def run_and_loss_trainable(
    trainable_params: NDArray,
    static_params: Sequence[float | None],
    quintuples: Sequence[Quintuple],
    q0: NDArray,
    transition_reward_func: Callable | None = None,
) -> float:
    """Evaluate loss in the reduced trainable subspace only."""
    params = materialize_params(trainable_params, static_params)
    return run_and_loss(params, quintuples, q0, transition_reward_func)


def fit(
    quintuples: list,
    q0: NDArray,
    p0: NDArray,
    static_params: list | None = None,
    transition_reward_func: Callable | None = None,
    user_param_bounds: Sequence[tuple[float | None, float | None]] = (),
    *,
    custom_param_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    fit_beta: bool = False,
    policy_beta: float = DEFAULT_POLICY_BETA,
) -> tuple[NDArray, float, NDArray, NDArray]:
    """Optimise SARSA parameters against observed quintuples.

    Parameters
    ----------
    quintuples : list of Quintuple
        Rollout transitions used for training.
    q0 : NDArray
        Initial Q-function prior to any updates.
    p0 : NDArray
        Initial guess for the full flat parameter vector.
    static_params : list[float | None] or None, optional
        Optional fixed parameter values matching the full parameter vector length.
        Explicit fixed values override the default beta hyperparameter handling.
    transition_reward_func : Callable or None, optional
        Optional callback with signature
        ``(user_params, s1, a1, s2) -> tuple[NDArray, float]``.
        When omitted, SARSA fits vanilla dynamics using observed rewards stored
        in ``quintuples``. When provided, the callback may recompute rewards from
        the user-defined parameter block.
    user_param_bounds : Sequence[tuple[float | None, float | None]], optional
        Bounds applied to user-defined parameters appended after the SARSA-owned
        parameter block. Leave empty for vanilla SARSA.
    custom_param_bounds : Sequence[tuple[float | None, float | None]] or None, optional
        Deprecated alias for ``user_param_bounds``. Kept temporarily for
        compatibility with the pre-refactor API.
    fit_beta : bool, optional
        If ``False`` (default), treat beta as a fixed policy hyperparameter.
        If ``True``, beta remains a trainable parameter unless explicitly fixed in
        ``static_params``.
    policy_beta : float, optional
        Fixed beta value used when ``fit_beta=False`` and ``static_params`` does
        not already provide a beta value.

    Returns
    -------
    NDArray
        Optimised parameter vector with static overrides applied.
    float
        Final loss value returned by the optimiser.
    NDArray
        Trajectory of Q-functions over the rollout.
    NDArray
        Probability of each action per timestep derived from the fitted policy.

    Raises
    ------
    AssertionError
        Propagated from :func:`run` if the reward callback returns an
        inconsistent next-state, or if logprob/action lengths mismatch.
    ValueError
        If quintuples are empty, indices are incompatible with ``q0``, parameter lengths
        are inconsistent, policy beta is invalid, or user-defined parameters are
        requested without a reward callback.

    Notes
    -----
    The underlying ``scipy.optimize.minimize`` may fail to converge.  Check the
    log output (INFO level) for the optimizer success flag and message.
    """
    p0 = np.asarray(p0, dtype=float)
    if custom_param_bounds is not None:
        if len(user_param_bounds) != 0:
            raise ValueError(
                "Specify only one of user_param_bounds or custom_param_bounds"
            )
        warnings.warn(
            "`custom_param_bounds` is deprecated; use `user_param_bounds` instead.",
            FutureWarning,
            stacklevel=2,
        )
        user_param_bounds = custom_param_bounds
    user_param_bounds = tuple(user_param_bounds)

    expected_param_count = len(SARSA_PARAM_BOUNDS) + len(user_param_bounds)
    if len(p0) != expected_param_count:
        raise ValueError(
            "p0 length must match SARSA plus user-defined parameters; "
            f"got {len(p0)} and expected {expected_param_count}"
        )
    if transition_reward_func is None and len(p0) != len(SARSA_PARAM_BOUNDS):
        raise ValueError("user-defined parameters require transition_reward_func")

    static_params = resolve_static_params(
        static_params,
        len(p0),
        fit_beta=fit_beta,
        policy_beta=policy_beta,
    )

    _validate_quintuples(quintuples, q0)
    full_bounds = SARSA_PARAM_BOUNDS + list(user_param_bounds)
    full_p0 = merge(p0, static_params)
    trainable_p0 = select_trainable_params(full_p0, static_params)
    trainable_bounds = select_trainable_bounds(full_bounds, static_params)

    if len(trainable_p0) == 0:
        params = full_p0
        loss = float(run_and_loss(params, quintuples, q0, transition_reward_func))
        logger.info("Optimizer skipped: all parameters fixed")
    else:
        res = optimize.minimize(
            run_and_loss_trainable,
            x0=trainable_p0,
            args=(static_params, quintuples, q0, transition_reward_func),
            bounds=trainable_bounds,
            method="L-BFGS-B",
        )

        loss = float(res.fun)
        params = materialize_params(res.x, static_params)
        logger.info(
            "Optimizer finished: success=%s, message=%s", res.success, res.message
        )

    q_trajectory, logprob_trajectory, error = run(
        params, quintuples, q0, transition_reward_func
    )

    action_prob = to_prob(logprob_trajectory)

    return params, loss, q_trajectory, action_prob
