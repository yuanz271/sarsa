from pathlib import Path

import numpy as np
import pandas as pd

from experiment import (
    Location,
    StateAxis,
    downsample_behavior_data,
    process_data,
    row_to_state,
    rt_lc_unp_state_spec,
)
import sarsa

# >>> Constants
SEED = 0
MIN_PENALTY = 1.0  # minimum shock penalty
REWARD_VALUE = 1.0  # value for liquid reward
CUSTOM_PARAM_BOUNDS = [
    (MIN_PENALTY, None),  # shock
    (0.0, None),  # avoidance
]


STATE_SPEC = rt_lc_unp_state_spec
# A finite state set is usually not flat but Cartesian product of many independent contexts, e.g.
# location, light and tone.
# Therefore the state specification is the tuple of the sizes of every context, e.g.
# (3, 2, 4) indicates 3 locations, 2 light status (on/off), and 4 tone status (off, 0-15s, 15-25s, 25-30s)
# Users can define their own states.
ACTION_SIZE = 3  # size of action set


def calc_reward(params, state):
    """Calculate the net reward of a state
    :param params: model parameters
    :param state: state

    To estimate the reward parameters, the value has to be calculated inside the optimization.
    """
    reward_value = REWARD_VALUE
    shock_value = params[3]
    escape_value = params[4]
    val = 0.0

    if state[StateAxis.Loc] == Location.R and state[StateAxis.Light] > 0:
        val += reward_value

    if state[StateAxis.Tone] == 3:
        if state[StateAxis.Loc] == Location.P:
            # Successful avoidance!!!
            val += escape_value
        else:
            # Shock!!!
            val -= shock_value

    return val


def init_params(rng, bounds):
    """Initialize parameter vector
    :param rng: NumPy random number generator
    :param bounds: bounds of parameters. See scipy.optimize.minimize for details.
    """
    bmin = np.array([b[0] for b in bounds])
    p0 = bmin + 0.5 * rng.random(size=len(sarsa.Param) + 2)
    return p0


def run_session(path: Path, rng):
    """
    :param path: path to data file
    :param rng: NumPy random number generator
    """
    # >>> Load data
    behavior_data = pd.read_csv(
        path,
        encoding="unicode_escape",
        header=0,
    )

    # rename first column to 'Time (s)'
    behavior_data = behavior_data.rename(columns={behavior_data.columns[0]: "Time (s)"})
    behavior_data.columns = map(str.upper, behavior_data.columns)
    behavior_data = downsample_behavior_data(behavior_data, "1s")
    behavior_data = process_data(behavior_data, "LC")
    # <<<

    # >>> Make quintuples
    quintuples = []
    T = len(behavior_data)  # RT:2500, LC: 3000, UNP: 3000
    for t in range(T - 2):
        # rows
        t1 = behavior_data.iloc[t]
        t2 = behavior_data.iloc[t + 1]
        t3 = behavior_data.iloc[t + 2]  # for finding the action at t+1

        # State:
        # s[t]: Array(3)
        # s[t,0]: location 0:P, 1:C, 2:R
        # s[t,1]: light 0, 1, (and 2, 3 for UNP)
        # s[t,2]: tone 0, 1, 2, 3

        s1 = row_to_state(t1)
        s2 = row_to_state(t2)
        s3 = row_to_state(t3)
        a1 = s2[StateAxis.Loc]
        a2 = s3[StateAxis.Loc]
        r2 = np.nan
        quintuples.append(sarsa.Quintuple(s1=s1, a1=a1, r2=r2, s2=s2, a2=a2))
    # <<<

    # >>> Prepare initial Q and parameters
    q0 = np.zeros((*STATE_SPEC, ACTION_SIZE))

    param_bounds = sarsa.PARAM_BOUNDS + CUSTOM_PARAM_BOUNDS
    p0 = init_params(rng, param_bounds)

    # >>> Fit
    params, loss, q_trajectory, action_prob = sarsa.fit(
        quintuples,
        q0=q0,
        p0=p0,
        static_params=None,
        reward_func=calc_reward,
        custom_param_bounds=CUSTOM_PARAM_BOUNDS,
    )

    # >>> Post-fit
    # q_trajectory: NDArray[T, *STATE_SPEC, ACTION_SIZE], list of Q functions every timestep
    # action_prob: NDArray[T, ACTION_SIZE], list of action probablity every timestep

    # Add your analysis code here
    # <<<


def main():
    run_session(Path.cwd() / "data" / "M1.csv", rng=np.random.default_rng(SEED))


if __name__ == "__main__":
    main()
