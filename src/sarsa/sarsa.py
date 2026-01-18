# Author: Yuan Zhao <yuan.zhao@nih.gov>
# Affiliation: Machine Learning Core, NIMH
"""
SARSA

This implementation is designed to be independent from the interpretation of state and action.
It only requires the state and action to be integer NumPy arrays.
It assumes that the first 3 components of parameter vector are alpha, beta and gamma.
The rest components are user-defined parameters (e.g. hidden reward)
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import log_softmax

EPS = 1e-8  # Minimum positive value
PARAM_BOUNDS = [
    (EPS, None),
    (EPS, None),
    (EPS, 1 - EPS),
]  # Bounds for SARSA parameters


class ParamIndex(IntEnum):
    """SARSA parameter index"""

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


def action_logprob(params, v) -> NDArray:
    """Compute softmax log-probabilities for each action.

    Parameters
    ----------
    params : NDArray
        Parameter vector with the inverse temperature stored at ``Param.beta``.
    v : NDArray
        Action-value estimates prior to scaling.

    Returns
    -------
    NDArray
        Log-probabilities over the action set after softmaxing ``v`` by ``beta``.
    """
    beta = params[ParamIndex.beta]
    return log_softmax(v * beta)


def to_prob(p):
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


def cross_entropy(inputs, targets):
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


def merge(params, static):
    """Combine trainable parameters with optional fixed values.

    Parameters
    ----------
    params : NDArray
        Candidate parameter values proposed by the optimiser.
    static : Sequence[float | None]
        Fixed values for each parameter position; ``None`` keeps the trainable value.

    Returns
    -------
    NDArray
        Parameter vector with static overrides applied.
    """
    return np.array(
        [p if s is None else s for p, s in zip(params, static)], dtype=float
    )


def update(params, quintuple: Quintuple, q: NDArray):
    """Apply the SARSA update for a single transition.

    Parameters
    ----------
    params : NDArray
        Parameter vector containing the learning rate and discount factor.
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
    alpha = params[ParamIndex.alpha]
    gamma = params[ParamIndex.gamma]
    q_new = q.copy()
    s1 = quintuple.s1
    a1 = quintuple.a1
    s2 = quintuple.s2
    a2 = quintuple.a2
    r = quintuple.r2

    error = r + gamma * q[*s2, a2] - q[*s1, a1]  # TD error

    q_new[*s1, a1] = q[*s1, a1] + alpha * error  # Update
    return q_new, error


def run(params, quintuples, q0, transition_reward_func):
    """Execute SARSA over a sequence of quintuples.

    Parameters
    ----------
    params : NDArray
        Parameter vector passed to the learning rule.
    quintuples : Sequence[Quintuple]
        Rollout transitions describing the trajectory to learn from.
    q0 : NDArray
        Initial Q-function prior to any updates.
    transition_reward_func : Callable
        Callback returning the new state and reward for a state and an action given the parameter vector.

    Returns
    -------
    NDArray
        Trajectory of Q-functions, including the initial state.
    NDArray
        Log-probabilities per timestep for the actions taken.
    NDArray
        Temporal-difference errors per timestep.
    """
    T = len(quintuples)
    qs = np.zeros((T + 1,) + q0.shape)
    error = np.zeros(T + 1)
    q = qs[0] = q0
    logprob = np.zeros((T, q0.shape[-1]))
    for t in range(T):
        quintuple = quintuples[t]
        logprob[t] = action_logprob(params, q[*quintuple.s1])
        s2, r2 = transition_reward_func(
            params,
            quintuple.s1,
            quintuple.a1,
            quintuple.s2,
        )  # calculate stepwise net reward on the fly for trainable reward-related parameters
        assert np.all(quintuple.s2 == s2)
        quintuple.r2 = r2
        qs[t + 1], error[t + 1] = update(params, quintuple, q)
        q = qs[t + 1]
    return qs, logprob, error


def run_and_loss(params, static, quintuples, q0, transition_reward_func):
    """Run SARSA and compute the cross-entropy loss.

    Parameters
    ----------
    params : NDArray
        Trainable parameter subset proposed by the optimiser.
    static : Sequence[float | None]
        Optional fixed parameter values to enforce during optimisation.
    quintuples : Sequence[Quintuple]
        Rollout transitions describing the trajectory to learn from.
    q0 : NDArray
        Initial Q-function prior to any updates.
    transition_reward_func : Callable
        Callback returning the new state and reward for a state and an action given the parameter vector.

    Returns
    -------
    float
        Mean cross-entropy loss between predicted and taken actions.
    """
    params = merge(
        params, static
    )  # transform parameters to constrained and replace with fixed values
    actions = np.array([q.a1 for q in quintuples], dtype=np.int_)
    q, logprob, _ = run(params, quintuples, q0, transition_reward_func)
    assert len(logprob) == len(actions), f"{len(logprob)}, {len(actions)}"
    ce = cross_entropy(logprob, actions)
    return ce


def fit(
    quintuples: list,
    q0: NDArray,
    p0: NDArray,
    static_params: list | None,
    transition_reward_func: Callable,
    custom_param_bounds,
):
    """Optimise SARSA parameters against observed quintuples.

    Parameters
    ----------
    quintuples : list of Quintuple
        Rollout transitions used for training.
    q0 : NDArray
        Initial Q-function prior to any updates.
    p0 : NDArray
        Initial guess for the optimiser across learnable parameters.
    static_params : list[float | None] or None
        Optional fixed parameter values, matching the length of ``p0`` plus custom parameters.
    transition_reward_func : Callable
        Callback returning the new state and reward for a state and an action given the parameter vector.
    custom_param_bounds : Sequence[tuple[float | None, float | None]]
        Bounds applied to custom parameters alongside the built-in SARSA bounds.

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
    """
    if static_params is None:
        static_params = [None] * len(p0)

    res = optimize.minimize(
        run_and_loss,
        x0=p0,
        args=(static_params, quintuples, q0, transition_reward_func),
        bounds=PARAM_BOUNDS + custom_param_bounds,
    )

    loss = res.fun  # type: ignore
    params = res.x  # type: ignore
    print(f"{res.success=}")
    params = merge(params, static_params)

    q_trajectory, logprob_trajectory, error = run(
        params, quintuples, q0, transition_reward_func
    )

    action_prob = to_prob(logprob_trajectory)

    return params, loss, q_trajectory, action_prob
