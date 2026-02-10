# Author: Yuan Zhao <yuan.zhao@nih.gov>
# Affiliation: Machine Learning Core, NIMH
"""
Integration test for SARSA fitting workflow.

Mirrors the workflow in examples/sarsa.ipynb to verify:
- fit() runs without error and optimizer converges
- Q-values update over time (sequential learning works)
- Loss is finite
"""

from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sarsa import sarsa

# ---------------------------------------------------------------------------
# Experiment helpers (copied from examples/experiment.py for test isolation)
#
# WARNING: These helpers are intentionally duplicated from examples/experiment.py
# to keep the test suite self-contained.  If the experiment logic changes in
# examples/experiment.py, the corresponding helpers here must be updated manually.
#
# Test data dependency: tests rely on examples/M1.csv (~6.3 MB behavioural
# dataset).  The file must be present for the test suite to pass.
# ---------------------------------------------------------------------------

LIGHT_ONSET_LC = np.array(
    [
        300,
        390,
        480,
        570,
        660,
        750,
        840,
        930,
        1020,
        1110,
        1200,
        1290,
        1380,
        1470,
        1560,
        1650,
        1740,
        1830,
        1920,
        2010,
        2100,
        2190,
        2280,
        2370,
        2460,
        2550,
        2640,
        2730,
        2820,
        2910,
    ],
    dtype=float,
)

TONE_ONSET_LC = np.array(
    [
        375,
        495,
        645,
        765,
        930,
        1035,
        1185,
        1320,
        1485,
        1590,
        1725,
        1830,
        1920,
        2085,
        2220,
        2295,
        2400,
        2565,
        2730,
        2895,
    ],
    dtype=float,
)

SHOCK_ONSET_LC = TONE_ONSET_LC + 28


class StateAxis(IntEnum):
    Loc = 0
    Light = 1
    Tone = 2


class Location(IntEnum):
    P = 0
    C = 1
    R = 2


STATE_SPEC = (3, 4, 4)  # 3 locations, 4 light states, 4 tone states
ACTION_SIZE = 3


def downsample_behavior_data(behavior_data, frequency):
    list_of_column_names = list(behavior_data.columns)
    behavior_data_ds = pd.DataFrame()

    for i in range(1, len(list_of_column_names)):
        col = list_of_column_names[i]
        if col in ("IN PLATFORM", "IN REWARD ZONE", "IN CENTER"):
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[col]
                .resample(frequency)
                .last()
            )
        elif col in ("NEW SPEAKER ACTIVE", "SHOCKER ON ACTIVE"):
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[col]
                .fillna(0)
                .resample(frequency)
                .last()
            )
        else:
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[col]
                .resample(frequency)
                .mean()
            )

        output.bfill(inplace=True)
        output.index = output.index.total_seconds()
        behavior_data_ds[col] = output

    return behavior_data_ds


def process_data(df):
    df = df[["IN PLATFORM", "IN CENTER", "IN REWARD ZONE"]]
    light_onset = pd.DataFrame(
        {"light_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in LIGHT_ONSET_LC:
        light_onset.loc[t : t + 9, "light_onset"] = 1
        light_onset.loc[t + 10 : t + 19, "light_onset"] = 2
        light_onset.loc[t + 20 : t + 29, "light_onset"] = 3

    tone_onset = pd.DataFrame(
        {"tone_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in TONE_ONSET_LC:
        tone_onset.loc[t : t + 14, "tone_onset"] = 1
        tone_onset.loc[t + 15 : t + 24, "tone_onset"] = 2
        tone_onset.loc[t + 25 : t + 29, "tone_onset"] = 3

    shock_onset = pd.DataFrame(
        {"shock_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in SHOCK_ONSET_LC:
        shock_onset.loc[t - 3 : t + 1, "shock_onset"] = 1

    features = pd.concat([light_onset, tone_onset, shock_onset], axis=1)
    df2 = pd.concat(
        [df.reset_index(drop=True), features.reset_index(drop=True)], axis=1
    )
    df2.set_index(df.index, inplace=True)
    return df2


def row_to_state(row):
    s = np.zeros(3, dtype=int)
    if row["IN PLATFORM"] > 0:
        s[StateAxis.Loc] = Location.P
    elif row["IN REWARD ZONE"] > 0:
        s[StateAxis.Loc] = Location.R
    else:
        s[StateAxis.Loc] = Location.C

    s[StateAxis.Light] = row["light_onset"]
    s[StateAxis.Tone] = row["tone_onset"]
    return s


# ---------------------------------------------------------------------------
# Test constants and helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MIN_PENALTY = 1.0
REWARD_VALUE = 1.0
CUSTOM_PARAM_BOUNDS = [
    (MIN_PENALTY, None),  # shock
    (0.0, None),  # avoidance
]


def transition_reward(params, state, action, new_state):
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
    bmin = np.array([b[0] for b in bounds])
    p0 = bmin + 0.5 * rng.random(size=len(sarsa.ParamIndex) + 2)
    return p0


def make_quintuples(behavior_data):
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_path():
    return PROJECT_ROOT / "examples" / "M1.csv"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def behavior_data(data_path):
    df = pd.read_csv(data_path, encoding="unicode_escape", header=0)
    df = df.rename(columns={df.columns[0]: "Time (s)"})
    df.columns = list(map(str.upper, df.columns))
    df = downsample_behavior_data(df, "1s")
    df = process_data(df)
    return df


@pytest.fixture
def quintuples(behavior_data):
    return make_quintuples(behavior_data)


@pytest.fixture
def initial_q():
    return np.zeros((*STATE_SPEC, ACTION_SIZE))


@pytest.fixture
def initial_params(rng):
    param_bounds = sarsa.PARAM_BOUNDS + CUSTOM_PARAM_BOUNDS
    return init_params(rng, param_bounds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSarsaFit:
    def test_fit_completes(self, quintuples, initial_q, initial_params):
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
        _, _, q_trajectory, _ = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            custom_param_bounds=CUSTOM_PARAM_BOUNDS,
        )
        assert not np.allclose(q_trajectory[0], q_trajectory[-1])

    def test_action_prob_shape(self, quintuples, initial_q, initial_params):
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
    def test_run_updates_q_sequentially(self, quintuples, initial_q):
        params = np.array([0.5, 1.0, 0.9, 1.0, 0.5])
        qs, logprob, error = sarsa.run(params, quintuples, initial_q, transition_reward)
        assert not np.allclose(qs[0], qs[-1])
