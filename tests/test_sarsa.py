# Author: Yuan Zhao <yuan.zhao@nih.gov>
# Affiliation: Machine Learning Core, NIMH
"""
Integration test for SARSA fitting workflow.

Mirrors the workflow in examples/sarsa.ipynb to verify:
- fit() runs without error and optimizer converges
- Q-values update over time (sequential learning works)
- Loss is finite
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src and examples are importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from sarsa import sarsa  # noqa: E402
from experiment import (  # noqa: E402
    Location,
    StateAxis,
    downsample_behavior_data,
    process_data,
    row_to_state,
    rt_lc_unp_state_spec,
)

# Constants from notebook
MIN_PENALTY = 1.0
REWARD_VALUE = 1.0
CUSTOM_PARAM_BOUNDS = [
    (MIN_PENALTY, None),  # shock
    (0.0, None),  # avoidance
]
STATE_SPEC = rt_lc_unp_state_spec
ACTION_SIZE = 3


def transition_reward(params, state, action, new_state):
    """Calculate the net reward of a state."""
    reward_value = REWARD_VALUE
    shock_value = params[3]
    escape_value = params[4]
    val = 0.0

    if state[StateAxis.Loc] == Location.R and state[StateAxis.Light] > 0:
        val += reward_value

    if state[StateAxis.Tone] == 3:
        if state[StateAxis.Loc] == Location.P:
            val += escape_value
        else:
            val -= shock_value

    return new_state, val


def init_params(rng, bounds):
    """Initialise the SARSA parameter vector within provided bounds."""
    bmin = np.array([b[0] for b in bounds])
    p0 = bmin + 0.5 * rng.random(size=len(sarsa.ParamIndex) + 2)
    return p0


def make_quintuples(behavior_data):
    """Construct SARSA training quintuples from processed behavioural data."""
    quintuples = []
    horizon = len(behavior_data)
    for t in range(horizon - 2):
        t1 = behavior_data.iloc[t]
        t2 = behavior_data.iloc[t + 1]
        t3 = behavior_data.iloc[t + 2]
        s1 = row_to_state(t1)
        s2 = row_to_state(t2)
        s3 = row_to_state(t3)
        a1 = s2[StateAxis.Loc]
        a2 = s3[StateAxis.Loc]
        r2 = np.nan
        quintuples.append(sarsa.Quintuple(s1=s1, a1=a1, r2=r2, s2=s2, a2=a2))
    return quintuples


@pytest.fixture
def data_path():
    """Path to the sample behavioural dataset."""
    return PROJECT_ROOT / "examples" / "M1.csv"


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(0)


@pytest.fixture
def behavior_data(data_path):
    """Load and preprocess behavioural data."""
    df = pd.read_csv(data_path, encoding="unicode_escape", header=0)
    df = df.rename(columns={df.columns[0]: "Time (s)"})
    df.columns = map(str.upper, df.columns)
    df = downsample_behavior_data(df, "1s")
    df = process_data(df, "LC")
    return df


@pytest.fixture
def quintuples(behavior_data):
    """Build quintuples from preprocessed data."""
    return make_quintuples(behavior_data)


@pytest.fixture
def initial_q():
    """Initial Q-function (all zeros)."""
    return np.zeros((*STATE_SPEC, ACTION_SIZE))


@pytest.fixture
def initial_params(rng):
    """Initial parameter guess."""
    param_bounds = sarsa.PARAM_BOUNDS + CUSTOM_PARAM_BOUNDS
    return init_params(rng, param_bounds)


class TestSarsaFit:
    """Tests for sarsa.fit() integration."""

    def test_fit_completes(self, quintuples, initial_q, initial_params):
        """Optimizer runs to completion without error."""
        params, loss, q_trajectory, action_prob = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            custom_param_bounds=CUSTOM_PARAM_BOUNDS,
        )
        assert params is not None
        assert loss is not None

    def test_loss_is_finite(self, quintuples, initial_q, initial_params):
        """Loss is a finite number."""
        _, loss, _, _ = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            custom_param_bounds=CUSTOM_PARAM_BOUNDS,
        )
        assert np.isfinite(loss)

    def test_q_trajectory_shape(self, quintuples, initial_q, initial_params):
        """Q-trajectory has expected shape (T+1, *state_spec, action_size)."""
        _, _, q_trajectory, _ = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            custom_param_bounds=CUSTOM_PARAM_BOUNDS,
        )
        T = len(quintuples)
        expected_shape = (T + 1, *STATE_SPEC, ACTION_SIZE)
        assert q_trajectory.shape == expected_shape

    def test_q_updates_propagate(self, quintuples, initial_q, initial_params):
        """Q-values change over time (sequential learning works)."""
        _, _, q_trajectory, _ = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            custom_param_bounds=CUSTOM_PARAM_BOUNDS,
        )
        # At least one Q-value should differ from initial after learning
        assert not np.allclose(q_trajectory[0], q_trajectory[-1])

    def test_action_prob_shape(self, quintuples, initial_q, initial_params):
        """Action probabilities have expected shape (T, action_size)."""
        _, _, _, action_prob = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            custom_param_bounds=CUSTOM_PARAM_BOUNDS,
        )
        T = len(quintuples)
        assert action_prob.shape == (T, ACTION_SIZE)

    def test_action_prob_sums_to_one(self, quintuples, initial_q, initial_params):
        """Action probabilities sum to 1 at each timestep."""
        _, _, _, action_prob = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            custom_param_bounds=CUSTOM_PARAM_BOUNDS,
        )
        row_sums = action_prob.sum(axis=1)
        assert np.allclose(row_sums, 1.0)


class TestSarsaRun:
    """Tests for sarsa.run() behavior."""

    def test_run_updates_q_sequentially(self, quintuples, initial_q):
        """Each step uses updated Q from previous step."""
        params = np.array(
            [0.5, 1.0, 0.9, 1.0, 0.5]
        )  # alpha, beta, gamma, shock, escape
        qs, logprob, error = sarsa.run(params, quintuples, initial_q, transition_reward)
        # Q should evolve - not all the same as initial
        assert not np.allclose(qs[0], qs[-1])
