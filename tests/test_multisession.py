# Author: Yuan Zhao <yuan.zhao@nih.gov>
# Affiliation: Machine Learning Core, NIMH
"""
Tests for the multi-session pooled SARSA fitting layer.

Key invariants (see ``spec/02_multisession_pooling.md``):

- A single session, all-shared, ``gap_rule="carry"`` reduces exactly to
  :func:`sarsa.sarsa.fit`.
- Two sessions, all-shared, ``carry``, ``d=1`` thread Q identically to
  :func:`sarsa.sarsa.run` over the concatenated quintuple stream.
- ``gap_rule="reset"`` with one session equals :func:`sarsa.sarsa.fit`.
- A session-specific mask recovers distinct parameters across sessions.
"""

import numpy as np
import pytest

from sarsa import sarsa
from sarsa import multisession as ms

pytestmark = pytest.mark.filterwarnings(
    "ignore:Optimized SARSA parameters landed on bounds.*"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_session(rng, n_states=2, n_actions=2, length=20):
    """Build a random-but-valid single session of quintuples."""
    quints = []
    s_prev = np.array([rng.integers(n_states)], dtype=int)
    a_prev = int(rng.integers(n_actions))
    for _ in range(length):
        s_next = np.array([rng.integers(n_states)], dtype=int)
        a_next = int(rng.integers(n_actions))
        r = float(rng.random())
        quints.append(sarsa.Quintuple(s1=s_prev, a1=a_prev, r2=r, s2=s_next, a2=a_next))
        s_prev, a_prev = s_next, a_next
    return quints


@pytest.fixture
def q0():
    return np.zeros((2, 2))  # (n_states, n_actions)


@pytest.fixture
def p0():
    # alpha, beta, gamma
    return np.array([0.4, sarsa.DEFAULT_POLICY_BETA, 0.9])


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def test_scatter_gather_roundtrip():
    share_mask = [True, True, False]  # gamma session-specific
    n_sessions = 3
    # shared(2) + 3 * spec(1) = 5
    full = np.array([0.1, 0.2, 0.31, 0.32, 0.33])
    thetas = ms.scatter_session_params(full, share_mask, n_sessions)
    assert len(thetas) == 3
    # shared values identical across sessions
    assert all(t[0] == 0.1 and t[1] == 0.2 for t in thetas)
    # session-specific gamma differs
    assert [t[2] for t in thetas] == [0.31, 0.32, 0.33]
    # gather is the inverse
    np.testing.assert_allclose(ms.gather_session_params(thetas, share_mask), full)


def test_optimizer_vector_length():
    assert ms.optimizer_vector_length([True, True, True], 4) == 3
    assert ms.optimizer_vector_length([True, True, False], 3) == 2 + 3 * 1


def test_scatter_length_mismatch_raises():
    with pytest.raises(ValueError):
        ms.scatter_session_params(np.zeros(4), [True, True, True], 2)


# ---------------------------------------------------------------------------
# Invariant 1: single session, all-shared, carry == fit()
# ---------------------------------------------------------------------------


def test_single_session_reduces_to_fit(q0, p0):
    rng = np.random.default_rng(0)
    session = make_session(rng)

    params, loss, q_traj, action_prob = sarsa.fit(session, q0=q0, p0=p0)

    result = ms.fit_subject([session], q0=q0, p0=p0, share_mask=[True, True, True])

    np.testing.assert_allclose(result.session_params[0], params, atol=1e-8)
    assert result.loss == pytest.approx(loss, abs=1e-10)
    np.testing.assert_allclose(result.q_trajectories[0], q_traj, atol=1e-8)
    np.testing.assert_allclose(result.action_probs[0], action_prob, atol=1e-8)


# ---------------------------------------------------------------------------
# Invariant 2: two sessions, carry, d=1 == run() over concatenation
# ---------------------------------------------------------------------------


def test_stacking_equivalence_carry(q0):
    rng = np.random.default_rng(1)
    s1 = make_session(rng, length=15)
    s2 = make_session(rng, length=12)
    params = np.array([0.5, 3.0, 0.8])

    # Multi-session forward pass with carry, d=1
    qs_list, logprob_list, _ = ms.run_subject(
        params, [s1, s2], q0, [True, True, True], gap_rule="carry", gap_decay=1.0
    )

    # Single continuous run over the concatenated stream
    qs_cat, logprob_cat, _ = sarsa.run(params, s1 + s2, q0)

    # Session 1 Q-end must equal the start of session 2's trajectory.
    np.testing.assert_allclose(qs_list[0][-1], qs_list[1][0], atol=1e-12)
    # Concatenated log-probs match.
    np.testing.assert_allclose(
        np.concatenate(logprob_list, axis=0), logprob_cat, atol=1e-12
    )
    # Final Q matches the continuous run's final Q.
    np.testing.assert_allclose(qs_list[1][-1], qs_cat[-1], atol=1e-12)


def test_pooled_loss_matches_concatenated_run(q0):
    rng = np.random.default_rng(2)
    s1 = make_session(rng, length=10)
    s2 = make_session(rng, length=14)
    params = np.array([0.3, 4.0, 0.7])

    pooled = ms.run_and_loss_subject(
        params, [s1, s2], q0, [True, True, True], gap_rule="carry", gap_decay=1.0
    )
    continuous = sarsa.run_and_loss(params, s1 + s2, q0)
    assert pooled == pytest.approx(continuous, abs=1e-12)


# ---------------------------------------------------------------------------
# Invariant 3: reset with one session == fit()
# ---------------------------------------------------------------------------


def test_reset_single_session_reduces_to_fit(q0, p0):
    rng = np.random.default_rng(3)
    session = make_session(rng)
    params, loss, _, _ = sarsa.fit(session, q0=q0, p0=p0)
    result = ms.fit_subject(
        [session], q0=q0, p0=p0, share_mask=[True, True, True], gap_rule="reset"
    )
    np.testing.assert_allclose(result.session_params[0], params, atol=1e-8)
    assert result.loss == pytest.approx(loss, abs=1e-10)


# ---------------------------------------------------------------------------
# Gap rules
# ---------------------------------------------------------------------------


def test_gap_decay_zero_resets_q(q0):
    rng = np.random.default_rng(4)
    s1 = make_session(rng, length=8)
    s2 = make_session(rng, length=8)
    params = np.array([0.5, 3.0, 0.8])

    qs_decay, _, _ = ms.run_subject(
        params, [s1, s2], q0, [True, True, True], gap_rule="decay", gap_decay=0.0
    )
    qs_reset, _, _ = ms.run_subject(
        params, [s1, s2], q0, [True, True, True], gap_rule="reset"
    )
    # decay with d=0 starts session 2 from zeros, same as reset (q0 is zeros).
    np.testing.assert_allclose(qs_decay[1][0], qs_reset[1][0], atol=1e-12)
    np.testing.assert_allclose(qs_decay[1][0], q0, atol=1e-12)


def test_invalid_gap_rule_raises(q0, p0):
    with pytest.raises(ValueError):
        ms.fit_subject([[]], q0=q0, p0=p0, gap_rule="bogus")


def test_gap_decay_out_of_range_raises(q0, p0):
    rng = np.random.default_rng(5)
    session = make_session(rng)
    with pytest.raises(ValueError):
        ms.fit_subject([session], q0=q0, p0=p0, gap_rule="decay", gap_decay=1.5)


# ---------------------------------------------------------------------------
# Masking: session-specific parameter recovery
# ---------------------------------------------------------------------------


def test_session_specific_mask_allows_distinct_params(q0):
    rng = np.random.default_rng(6)
    # Two sessions; share alpha and gamma, but beta session-specific and fitted.
    s1 = make_session(rng, length=40)
    s2 = make_session(rng, length=40)
    share_mask = [True, False, True]  # beta session-specific
    p0 = np.array([0.4, 3.0, 0.9])

    result = ms.fit_subject(
        [s1, s2],
        q0=q0,
        p0=p0,
        share_mask=share_mask,
        fit_beta=True,
    )
    # shared alpha/gamma identical across sessions
    assert result.session_params[0][0] == pytest.approx(result.session_params[1][0])
    assert result.session_params[0][2] == pytest.approx(result.session_params[1][2])
    # beta is session-specific (independently fitted; allowed to differ)
    assert len(result.shared_params) == 2  # alpha, gamma shared


def test_empty_sessions_raises(q0, p0):
    with pytest.raises(ValueError):
        ms.fit_subject([], q0=q0, p0=p0)
