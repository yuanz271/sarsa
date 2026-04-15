# Author: Yuan Zhao <yuan.zhao@nih.gov>
# Affiliation: Machine Learning Core, NIMH
"""
Integration test for SARSA fitting workflow.

Verifies:
- fit() runs without error and optimizer converges
- Q-values update over time (sequential learning works)
- Loss is finite
"""

from enum import IntEnum
from importlib import metadata
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sarsa as sarsa_package
from sarsa import sarsa

pytestmark = pytest.mark.filterwarnings(
    "ignore:Optimized SARSA parameters landed on bounds.*"
)

# ---------------------------------------------------------------------------
# Synthetic experiment helpers for test isolation
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

MIN_PENALTY = 1.0
REWARD_VALUE = 1.0
USER_PARAM_BOUNDS = [
    (MIN_PENALTY, None),  # shock
    (0.0, None),  # avoidance
]


class UserParamIndex(IntEnum):
    shock = 0
    avoidance = 1


def transition_reward(user_params, state, action, new_state):
    reward_value = REWARD_VALUE
    shock_value = user_params[UserParamIndex.shock]
    escape_value = user_params[UserParamIndex.avoidance]
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
    p0 = bmin + 0.5 * rng.random(size=len(bounds))
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


def make_toy_quintuples(reward=2.0):
    return [
        sarsa.Quintuple(
            s1=np.array([0], dtype=int),
            a1=0,
            r2=reward,
            s2=np.array([0], dtype=int),
            a2=0,
        ),
        sarsa.Quintuple(
            s1=np.array([0], dtype=int),
            a1=0,
            r2=reward,
            s2=np.array([0], dtype=int),
            a2=0,
        ),
    ]


def toy_transition_reward(user_params, state, action, new_state):
    return new_state, user_params[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def behavior_data():
    horizon = int(SHOCK_ONSET_LC.max() + 40)
    time = np.arange(horizon, dtype=float)

    loc = np.arange(horizon, dtype=int) % 3
    in_platform = (loc == Location.P).astype(int)
    in_center = (loc == Location.C).astype(int)
    in_reward_zone = (loc == Location.R).astype(int)

    df = pd.DataFrame(
        {
            "TIME (S)": time,
            "IN PLATFORM": in_platform,
            "IN CENTER": in_center,
            "IN REWARD ZONE": in_reward_zone,
            "NEW SPEAKER ACTIVE": np.zeros(horizon, dtype=int),
            "SHOCKER ON ACTIVE": np.zeros(horizon, dtype=int),
        }
    )

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
    param_bounds = sarsa.SARSA_PARAM_BOUNDS + USER_PARAM_BOUNDS
    return init_params(rng, param_bounds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPackageMetadata:
    def test_package_version_matches_installed_metadata(self):
        assert sarsa_package.__version__ == metadata.version("sarsa")


class TestParameterPacking:
    def test_concat_and_split_roundtrip(self):
        sarsa_params = np.array([0.5, 1.0, 0.9])
        user_params = np.array([1.5, 0.25])

        params = sarsa.concat_params(sarsa_params, user_params)
        split_sarsa_params, split_user_params = sarsa.split_params(params)

        assert np.allclose(split_sarsa_params, sarsa_params)
        assert np.allclose(split_user_params, user_params)


class TestSarsaParameterBounds:
    def test_canonical_bounds_match_edge_safe_domains(self):
        assert sarsa.SARSA_PARAM_BOUNDS == [
            (0.0, 1.0),
            (0.0, None),
            (0.0, 1.0 - sarsa.EPS),
        ]

    def test_update_allows_alpha_zero_without_learning(self):
        quintuple = make_toy_quintuples(reward=2.0)[0]
        q0 = np.zeros((1, 2))

        q_new, error = sarsa.update(np.array([0.0, 1.0, 0.5]), quintuple, q0)

        assert np.allclose(q_new, q0)
        assert error == pytest.approx(2.0)

    def test_update_allows_alpha_one_full_td_step(self):
        quintuple = make_toy_quintuples(reward=2.0)[0]
        q0 = np.array([[1.0, 0.0]])

        q_new, error = sarsa.update(np.array([1.0, 1.0, 0.5]), quintuple, q0)

        assert error == pytest.approx(1.5)
        assert q_new[0, 0] == pytest.approx(2.5)

    def test_update_allows_gamma_zero_for_immediate_reward_only(self):
        quintuple = make_toy_quintuples(reward=2.0)[0]
        q0 = np.array([[1.0, 0.0]])

        q_new, error = sarsa.update(np.array([1.0, 1.0, 0.0]), quintuple, q0)

        assert error == pytest.approx(1.0)
        assert q_new[0, 0] == pytest.approx(2.0)

    def test_action_logprob_is_uniform_when_beta_is_zero(self):
        logprob = sarsa.action_logprob(
            np.array([0.5, 0.0, 0.5]),
            np.array([2.0, -1.0, 7.0]),
        )

        assert np.allclose(sarsa.to_prob(logprob), np.full(3, 1 / 3))


class TestSarsaFit:
    def test_fit_completes(self, quintuples, initial_q, initial_params):
        params, loss, q_trajectory, action_prob = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            user_param_bounds=USER_PARAM_BOUNDS,
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
            user_param_bounds=USER_PARAM_BOUNDS,
        )
        assert np.isfinite(loss)

    def test_q_trajectory_shape(self, quintuples, initial_q, initial_params):
        _, _, q_trajectory, _ = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            user_param_bounds=USER_PARAM_BOUNDS,
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
            user_param_bounds=USER_PARAM_BOUNDS,
        )
        assert not np.allclose(q_trajectory[0], q_trajectory[-1])

    def test_action_prob_shape(self, quintuples, initial_q, initial_params):
        _, _, _, action_prob = sarsa.fit(
            quintuples,
            q0=initial_q,
            p0=initial_params,
            static_params=None,
            transition_reward_func=transition_reward,
            user_param_bounds=USER_PARAM_BOUNDS,
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
            user_param_bounds=USER_PARAM_BOUNDS,
        )
        row_sums = action_prob.sum(axis=1)
        assert np.allclose(row_sums, 1.0)


class TestSarsaRun:
    def test_run_updates_q_sequentially(self, quintuples, initial_q):
        params = sarsa.concat_params(np.array([0.5, 1.0, 0.9]), np.array([1.0, 0.5]))
        qs, logprob, error = sarsa.run(params, quintuples, initial_q, transition_reward)
        assert not np.allclose(qs[0], qs[-1])

    def test_run_uses_observed_rewards_in_vanilla_mode(self):
        quintuples = make_toy_quintuples(reward=2.0)
        q0 = np.zeros((1, 2))
        params = np.array([1.0, 1.0, 0.0])

        qs, _, error = sarsa.run(params, quintuples, q0)

        assert qs[1, 0, 0] == pytest.approx(2.0)
        assert error[0] == pytest.approx(2.0)
        assert quintuples[0].r2 == pytest.approx(2.0)

    def test_run_passes_only_user_params_to_reward_callback(self):
        quintuples = make_toy_quintuples(reward=2.0)
        q0 = np.zeros((1, 2))
        params = sarsa.concat_params(np.array([1.0, 1.0, 0.0]), np.array([5.0, 7.0]))
        captured = {}

        def record_user_params(user_params, state, action, new_state):
            captured["user_params"] = user_params.copy()
            return new_state, user_params[0]

        sarsa.run(params, quintuples, q0, record_user_params)

        assert np.allclose(captured["user_params"], np.array([5.0, 7.0]))

    def test_run_recomputes_rewards_when_callback_is_provided(self):
        quintuples = make_toy_quintuples(reward=2.0)
        q0 = np.zeros((1, 2))
        params = sarsa.concat_params(np.array([1.0, 1.0, 0.0]), np.array([5.0]))

        qs, _, error = sarsa.run(params, quintuples, q0, toy_transition_reward)

        assert qs[1, 0, 0] == pytest.approx(5.0)
        assert error[0] == pytest.approx(5.0)
        assert quintuples[0].r2 == pytest.approx(2.0)

    def test_run_rejects_missing_rewards_in_vanilla_mode(self):
        quintuples = make_toy_quintuples(reward=np.nan)
        q0 = np.zeros((1, 2))
        params = np.array([1.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="finite observed rewards"):
            sarsa.run(params, quintuples, q0)


class TestSarsaFitVanilla:
    def test_fit_completes_without_reward_callback(self):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.5, 1.0, 0.5])

        params, loss, q_trajectory, action_prob = sarsa.fit(quintuples, q0, p0)

        assert params.shape == (len(sarsa.SARSA_PARAM_BOUNDS),)
        assert np.isfinite(loss)
        assert q_trajectory.shape == (len(quintuples) + 1, 1, 2)
        assert action_prob.shape == (len(quintuples), 2)
        assert params[sarsa.ParamIndex.beta] == pytest.approx(sarsa.DEFAULT_POLICY_BETA)

    def test_fit_rejects_extra_params_without_reward_callback(self):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.5, 1.0, 0.5, 2.0])

        with pytest.raises(ValueError, match="user-defined parameters require"):
            sarsa.fit(quintuples, q0, p0, user_param_bounds=[(0.0, None)])


class TestSarsaFitOptimization:
    def test_fit_fixes_beta_by_default_and_uses_reduced_subspace(self, monkeypatch):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.1, 0.5])
        captured = {}

        def fake_minimize(fun, x0, args, bounds, method):
            captured["x0"] = np.array(x0, copy=True)
            captured["bounds"] = list(bounds)
            captured["method"] = method
            return SimpleNamespace(x=np.array(x0, copy=True), fun=0.0, success=True, message="ok")

        monkeypatch.setattr(sarsa.optimize, "minimize", fake_minimize)

        params, loss, _, _ = sarsa.fit(quintuples, q0, p0)

        assert captured["method"] == "L-BFGS-B"
        assert captured["x0"].shape == (2,)
        assert captured["bounds"] == [
            sarsa.SARSA_PARAM_BOUNDS[sarsa.ParamIndex.alpha],
            sarsa.SARSA_PARAM_BOUNDS[sarsa.ParamIndex.gamma],
        ]
        assert loss == pytest.approx(0.0)
        assert params[sarsa.ParamIndex.alpha] == pytest.approx(p0[sarsa.ParamIndex.alpha])
        assert params[sarsa.ParamIndex.beta] == pytest.approx(sarsa.DEFAULT_POLICY_BETA)
        assert params[sarsa.ParamIndex.gamma] == pytest.approx(p0[sarsa.ParamIndex.gamma])

    def test_fit_can_opt_in_to_beta_optimization(self, monkeypatch):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.75, 0.5])
        captured = {}

        def fake_minimize(fun, x0, args, bounds, method):
            captured["x0"] = np.array(x0, copy=True)
            captured["bounds"] = list(bounds)
            captured["method"] = method
            return SimpleNamespace(x=np.array(x0, copy=True), fun=0.0, success=True, message="ok")

        monkeypatch.setattr(sarsa.optimize, "minimize", fake_minimize)

        params, _, _, _ = sarsa.fit(quintuples, q0, p0, fit_beta=True)

        assert captured["method"] == "L-BFGS-B"
        assert captured["x0"].shape == (3,)
        assert captured["bounds"] == list(sarsa.SARSA_PARAM_BOUNDS)
        assert params[sarsa.ParamIndex.beta] == pytest.approx(p0[sarsa.ParamIndex.beta])

    def test_explicit_static_beta_overrides_fit_beta(self, monkeypatch):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.75, 0.5])
        captured = {}

        def fake_minimize(fun, x0, args, bounds, method):
            captured["x0"] = np.array(x0, copy=True)
            return SimpleNamespace(x=np.array(x0, copy=True), fun=0.0, success=True, message="ok")

        monkeypatch.setattr(sarsa.optimize, "minimize", fake_minimize)

        with pytest.warns(UserWarning, match="fit_beta=True ignored"):
            params, _, _, _ = sarsa.fit(
                quintuples,
                q0,
                p0,
                static_params=[None, 2.5, None],
                fit_beta=True,
            )

        assert captured["x0"].shape == (2,)
        assert params[sarsa.ParamIndex.beta] == pytest.approx(2.5)

    def test_fit_skips_optimizer_when_all_params_are_fixed(self, monkeypatch):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.75, 0.5])

        def fail_minimize(*args, **kwargs):
            raise AssertionError("optimizer should not be called when all params are fixed")

        monkeypatch.setattr(sarsa.optimize, "minimize", fail_minimize)

        params, loss, q_trajectory, action_prob = sarsa.fit(
            quintuples,
            q0,
            p0,
            static_params=[0.25, 2.5, 0.5],
        )

        assert np.allclose(params, np.array([0.25, 2.5, 0.5]))
        assert np.isfinite(loss)
        assert q_trajectory.shape == (len(quintuples) + 1, 1, 2)
        assert action_prob.shape == (len(quintuples), 2)

    def test_fit_accepts_zero_policy_beta(self, monkeypatch):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.75, 0.5])
        captured = {}

        def fake_minimize(fun, x0, args, bounds, method):
            captured["bounds"] = list(bounds)
            return SimpleNamespace(x=np.array(x0, copy=True), fun=0.0, success=True, message="ok")

        monkeypatch.setattr(sarsa.optimize, "minimize", fake_minimize)

        params, loss, _, _ = sarsa.fit(quintuples, q0, p0, policy_beta=0.0)

        assert captured["bounds"] == [
            sarsa.SARSA_PARAM_BOUNDS[sarsa.ParamIndex.alpha],
            sarsa.SARSA_PARAM_BOUNDS[sarsa.ParamIndex.gamma],
        ]
        assert loss == pytest.approx(0.0)
        assert params[sarsa.ParamIndex.beta] == pytest.approx(0.0)

    def test_fit_rejects_negative_policy_beta(self):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.75, 0.5])

        with pytest.raises(ValueError, match="non-negative"):
            sarsa.fit(quintuples, q0, p0, policy_beta=-0.1)

    def test_fit_accepts_static_sarsa_edge_values(self, monkeypatch):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.75, 0.5])

        def fail_minimize(*args, **kwargs):
            raise AssertionError("optimizer should not be called when all params are fixed")

        monkeypatch.setattr(sarsa.optimize, "minimize", fail_minimize)

        params, loss, _, _ = sarsa.fit(
            quintuples,
            q0,
            p0,
            static_params=[0.0, 0.0, 1.0 - sarsa.EPS],
        )

        assert np.allclose(params, np.array([0.0, 0.0, 1.0 - sarsa.EPS]))
        assert np.isfinite(loss)

    @pytest.mark.parametrize(
        "static_params",
        [
            [1.5, None, None],
            [None, -0.1, None],
            [None, None, 1.0],
        ],
    )
    def test_fit_rejects_out_of_bounds_static_sarsa_params(self, static_params):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.75, 0.5])

        with pytest.raises(ValueError, match=r"static_params\[[0-2]\].*violates bounds"):
            sarsa.fit(quintuples, q0, p0, static_params=static_params)

    def test_fit_rejects_out_of_bounds_static_user_param(self):
        quintuples = make_toy_quintuples(reward=np.nan)
        q0 = np.zeros((1, 2))
        p0 = sarsa.concat_params(np.array([0.25, 0.75, 0.5]), np.array([2.0]))

        with pytest.raises(ValueError, match=r"static_params\[3\].*violates bounds"):
            sarsa.fit(
                quintuples,
                q0,
                p0,
                static_params=[None, None, None, 0.5],
                transition_reward_func=toy_transition_reward,
                user_param_bounds=[(1.0, None)],
            )

    def test_fit_warns_when_trainable_sarsa_param_hits_bound(self, monkeypatch):
        quintuples = make_toy_quintuples(reward=1.0)
        q0 = np.zeros((1, 2))
        p0 = np.array([0.25, 0.75, 0.5])

        def fake_minimize(fun, x0, args, bounds, method):
            return SimpleNamespace(
                x=np.array([x0[0], 1.0 - sarsa.EPS]),
                fun=0.0,
                success=True,
                message="ok",
            )

        monkeypatch.setattr(sarsa.optimize, "minimize", fake_minimize)

        with pytest.warns(UserWarning, match=r"landed on bounds.*gamma≈upper"):
            params, _, _, _ = sarsa.fit(quintuples, q0, p0)

        assert params[sarsa.ParamIndex.gamma] == pytest.approx(1.0 - sarsa.EPS)


class TestSarsaFitCompatibility:
    def test_fit_accepts_deprecated_custom_param_bounds_alias(self):
        quintuples = make_toy_quintuples(reward=np.nan)
        q0 = np.zeros((1, 2))
        p0 = sarsa.concat_params(np.array([0.5, 1.0, 0.5]), np.array([2.0]))

        with pytest.warns(FutureWarning, match="custom_param_bounds"):
            params, loss, q_trajectory, action_prob = sarsa.fit(
                quintuples,
                q0,
                p0,
                transition_reward_func=toy_transition_reward,
                custom_param_bounds=[(0.0, None)],
            )

        assert params.shape == (4,)
        assert np.isfinite(loss)
        assert q_trajectory.shape == (len(quintuples) + 1, 1, 2)
        assert action_prob.shape == (len(quintuples), 2)

    def test_fit_rejects_both_user_and_custom_param_bounds(self):
        quintuples = make_toy_quintuples(reward=np.nan)
        q0 = np.zeros((1, 2))
        p0 = sarsa.concat_params(np.array([0.5, 1.0, 0.5]), np.array([2.0]))

        with pytest.raises(
            ValueError, match="Specify only one of user_param_bounds or custom_param_bounds"
        ):
            sarsa.fit(
                quintuples,
                q0,
                p0,
                transition_reward_func=toy_transition_reward,
                user_param_bounds=[(0.0, None)],
                custom_param_bounds=[(0.0, None)],
            )
