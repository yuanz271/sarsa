# SARSA Algorithm

SARSA is an on-policy temporal-difference (TD) reinforcement learning algorithm.
It learns an action-value function **Q(s, a)** directly from experience by updating
predictions step-by-step as transitions are observed.

In this implementation, **state is composite** and **action is flat**:
- `s` is represented as a length-`k` integer vector of discrete state-factor
  indices, one entry per state factor
- `a` is represented as a single integer index into the action axis

Equivalently, `Q` has shape `(*state_dims, n_actions)`.
For example, if `Q.shape == (3, 4, 4, 3)`, then `s = [2, 1, 3]` and `a = 0`
index `Q[2, 1, 3, 0]`.

## Action-value function

Q(s, a) is the expected cumulative discounted reward when taking action **a** in
state **s** and following the current policy thereafter:

$$Q(s_t, a_t) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t, a_t\right]$$

## TD update rule

At each timestep, SARSA observes the transition **(s₁, a₁, r₂, s₂, a₂)** and updates:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \cdot \delta_t$$

where the **TD error** δ is:

$$\delta_t = r_{t+1} + \gamma \cdot Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$$

## Parameters

| Parameter | Symbol | Role | Bounds |
|-----------|--------|------|--------|
| Learning rate | α | Controls how fast Q-values update | (0, 1] |
| Inverse temperature | β | Controls action selection determinism via softmax | [0, MAX_BETA] |
| Discount factor | γ | Controls how much future rewards are valued | [0, 1) |

In the current implementation, these live in the **SARSA-owned parameter block**:

```python
sarsa_params = (alpha, beta, gamma)
```

The canonical role of `beta` is unchanged, but the current `fit()` API treats it
as a fixed policy hyperparameter by default. In practice, fitting usually targets
`alpha`, `gamma`, and any `user_params`, while `beta` is held at
`sarsa.DEFAULT_POLICY_BETA` unless `fit_beta=True` is requested.

The implementation excludes the degenerate, likelihood-flattening edge
`alpha = 0` (no learning) via a small positive lower bound, and caps `beta` at
`MAX_BETA` (= 20) to avoid the weakly-identified near-deterministic regime. The
meaningful boundary cases `alpha = 1` (full TD step), `beta = 0` (uniform
policy), and `gamma = 0` (myopic) are allowed; only exact `gamma = 1` is
excluded, since that changes the generic discounted-return setup.

## Policy

Actions are selected according to a **softmax policy**:

$$P(a \mid s) = \frac{\exp(\beta \cdot Q(s, a))}{\sum_{a'} \exp(\beta \cdot Q(s, a'))}$$

- High β → deterministic (exploit)
- Low β → random (explore)

This makes `beta` a natural policy-readout hyperparameter when the main inferential
target is the learning dynamics rather than the action-selection temperature itself.

## Algorithm

```
initialise Q(s, a) = 0 for all s, a

for each transition (s1, a1, r2, s2, a2):
    delta = r2 + gamma * Q(s2, a2) - Q(s1, a1)
    Q(s1, a1) = Q(s1, a1) + alpha * delta
```

## On-policy vs off-policy

SARSA is **on-policy**: the update uses the action **actually taken** next (`a2`),
not the best possible action. This distinguishes it from Q-learning, which uses
`max_a Q(s2, a)`.

## Extension: user-defined parameters

This implementation supports an additional **user-defined parameter block** that is
concatenated after the SARSA block in the flat optimizer vector:

```python
params = sarsa.concat_params(sarsa_params, user_params)
```

The vanilla SARSA kernel still canonically interprets only `alpha`, `beta`, and `gamma`.
User-defined parameters are passed to extension callbacks such as
`transition_reward_func(user_params, s1, a1, s2)`.

> **API note:** As of `v0.3.0`, `transition_reward_func` receives `user_params`.
> Before `v0.3.0`, the reward callback received the full flat parameter vector.

See [Manual](manual.md) for the full API and [Example](example.md) for a worked example.
