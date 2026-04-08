---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 24px;
  }
  h2 {
    font-size: 34px;
    margin-bottom: 0.2em;
  }
  h4 {
    font-size: 24px;
    margin-bottom: 0.1em;
    margin-top: 0.3em;
  }
  ul, ol {
    margin: 0.2em 0;
  }
  li {
    margin: 0.15em 0;
    line-height: 1.3;
  }
  p {
    margin: 0.2em 0;
  }
  p.cite {
    text-align: right;
    font-size: 18px;
    color: #888;
    margin-top: 0.5em;
  }
---

# The Neural Signature of Prediction Error and SARSA

How Dopamine Implements SARSA Learning in the Brain

---

## Reinforcement Learning (RL)

- An **agent** interacts with an **environment** over time
- At each step: observe state **s**, take action **a**, receive reward **r**
- Goal: learn a **policy** π(a|s) that maximizes cumulative future reward

#### Key concepts

- **Policy:** how the agent selects actions given a state
- **Reward:** scalar feedback signal from the environment
- **Value:** expected cumulative reward from a state (or state-action pair, **Q**)

<p class="cite">Sutton & Barto (2018)</p>

---

## Q: Expected Cumulative Reward

Q(s, a) is the **action-value function** — expected cumulative discounted reward when taking action **a** in state **s** and following the current policy thereafter:

$$Q(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1} \middle| s_0 = s, a_0 = a\right]$$

*"How good is it to take action a from state s?"*

- **s** — current state (sensory context, task state)
- **a** — action taken
- **r** — reward received after the action
- **γ** — discount factor (how much future rewards are worth)

<p class="cite">Sutton & Barto (2018)</p>

---

## TD Learning and SARSA

- **Temporal-Difference (TD) learning:** Learn from experience by updating predictions step-by-step
- **SARSA:** On-policy TD algorithm — updates Q using the action **actually taken** next (not the best possible action)

#### SARSA Update

- Q(s₁, a₁) ← Q(s₁, a₁) + α · [r₂ + γ·Q(s₂, a₂) − Q(s₁, a₁)]

#### Key Parameters

- **α** (alpha) — learning rate: how fast Q-values update
- **β** (beta) — inverse temperature: how deterministic action selection is
- **γ** (gamma) — discount factor: how much future rewards are valued

#### Core Question

- What do these parameters correspond to **in the brain**?

<p class="cite">Sutton & Barto (2018)</p>

---

## Prediction Error (PE)

- Definition: Discrepancy between expected outcome and actual outcome
- Since Q is cumulative: Q(s₁, a₁) - γ·Q(s₂, a₂) = predicted reward at step 1
- PE is the residual: **δ = r₂ - [Q(s₁, a₁) - γ·Q(s₂, a₂)]** = r₂ + γ·Q(s₂, a₂) - Q(s₁, a₁)
- Acts as a learning signal

#### Dopamine Responses

- Positive PE (reward > expected): Dopamine burst ↑
- Zero PE (reward = expected): Dopamine baseline
- Negative PE (reward < expected): Dopamine pause ↓

<p class="cite">Schultz et al. (1997)</p>

---

## Parameter α - Learning Rate

- **Controls:** Q ← Q + α·δ
- **High α:** Large updates, fast learning
- **Low α:** Small updates, stable learning

#### Neural Substrate

- High α: Strong dopamine → rapid synaptic potentiation → fast learning
- Low α: Weak dopamine → slow plasticity → stable learning

#### Observable Correlates

- Correlates with dopamine receptor (D1/D2) density
- High-α subjects show faster behavioral adaptation
- Varies by genetics, neuromodulator state, arousal

<p class="cite">Reynolds & Wickens (2002); Wise (2004)</p>

---

## Parameter γ - Discount Factor

- **High γ (0.9–0.99):** Values distant future → long-horizon planning
- **Low γ (0.1–0.5):** Myopic, immediate-reward focused

#### Neural Substrate

- Striatum predicts reward timing
- Enables ventral tegmental area (VTA) to compute γ·Q(s₂, a₂)
- Ventral striatum lesions disrupt timing (not quantity) coding

#### Temporal Shift Phenomenon

- Early learning: Dopamine bursts at reward
- After learning: Dopamine shifts backward to predictive cue
- Distance of shift ∝ γ (higher γ = larger shift)

<p class="cite">Pan et al. (2005); Hollerman & Schultz (1998)</p>

---

## Parameter β - Inverse Temperature

- **Controls:** P(action | state) = softmax(β·Q(s, :))
- **High β:** Deterministic choices (exploit)
- **Low β:** Random exploration

#### Neural Substrate

- High dopamine tone → high β (confident, exploitative)
- Low dopamine tone → low β (uncertain, exploratory)

#### Observable Correlates

- Correlates with tonic dopamine level
- Increases with task engagement/arousal
- Decreases with fatigue, stress, uncertainty

<p class="cite">Daw (2011); Humphries et al. (2012)</p>

---

## Relevant Brain Regions

- **Ventral Tegmental Area (VTA):** Dopamine generator; computes reward prediction error (RPE); broadcasts to striatum, amygdala, prefrontal cortex
- **Substantia Nigra pars compacta (SNc):** Secondary dopamine source; encodes RPE; motor control
- **Nucleus Accumbens (NAc):** PE receiver; encodes action values; reward timing
- **Dorsal Striatum:** Integrates sensory + reward PE; guides decisions
- **Prefrontal Cortex & Hippocampus:** Planning and memory updates

<p class="cite">Schultz (2015)</p>

---

## The VTA→Striatum Circuit

```
SENSORY INPUT
      ↓
STRIATUM (encodes action values, selects action)   [? interpretive]
      ↓
OUTCOME: Reward r₂, Next State s₂
      ↓
VTA DOPAMINE: computes RPE ≈ δ                    [✓ Schultz 1997]
      ↓
DOPAMINE RELEASE → STRIATUM (~50 ms onset)        [✓ Hollerman & Schultz 1998]
      ↓
SYNAPTIC PLASTICITY: ΔW ∝ δ · eligibility(t)     [✓ Reynolds 2002]
      ↓
ACTION VALUE UPDATE                               [? interpretive]
      ↓  (loop)
```

- NAc → VTA feedback: GABAergic inhibitory  [✓ Xia et al. 2011]
- Feedback encodes value predictions         [? not established]

<p class="cite">Schultz et al. (1997); Montague et al. (1996)</p>

---

## SARSA as a Window into the Brain

- SARSA formalizes learning from prediction errors using three parameters:
  - **α** ↔ dopamine-dependent synaptic plasticity
  - **γ** ↔ temporal credit assignment in striatum
  - **β** ↔ tonic dopamine and exploration-exploitation
- Dopamine firing is consistent with TD error [Schultz et al. 1997]
- SARSA is a **parsimonious proxy** — not the brain's algorithm, but a useful lens
- Open questions: continuous vs. discrete time, exact credit assignment, alternative RL models
- Fitting SARSA to behavior may offer **approximate neural readouts**

---

## References (1/2)

1. Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning. In *Classical Conditioning II: Current Research and Theory* (pp. 64–99). Appleton-Century-Crofts.

2. Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593–1599.

3. Montague, P. R., Dayan, P., & Sejnowski, T. J. (1996). A framework for mesencephalic dopamine systems based on predictive Hebbian learning. *Journal of Neuroscience*, 16(5), 1936–1947.

4. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

5. Wise, R. A. (2004). Dopamine, learning and motivation. *Nature Reviews Neuroscience*, 5(6), 483–494.

6. Reynolds, J. N. J., & Wickens, J. R. (2002). Dopamine-dependent plasticity of corticostriatal synapses. *Neural Networks*, 15(4–6), 507–521.

---

## References (2/2)

7. Pan, W. X., Schmidt, R., Wickens, J. R., & Hyland, B. I. (2005). Dopamine cells respond to predicted events during classical conditioning. *Journal of Neuroscience*, 25(26), 6235–6242.

8. Schultz, W. (2015). Neuronal reward and decision signals: From theories to data. *Physiological Reviews*, 95(3), 853–951.

9. Hollerman, J. R., & Schultz, W. (1998). Dopamine neurons report an error in the temporal prediction of reward during learning. *Nature Neuroscience*, 1(4), 304–309.

10. Daw, N. D. (2011). Trial-by-trial data analysis using computational models. *Decision Making, Affect, and Learning*, 3–38.

11. Humphries, M. D., Khamassi, M., & Gurney, K. (2012). Dopaminergic control of the exploration-exploitation trade-off via the basal ganglia. *Frontiers in Neuroscience*, 6, 9.

12. Xia, Y., Driscoll, J. R., Wilbrecht, L., Margolis, E. B., Fields, H. L., & Hjelmstad, G. O. (2011). Nucleus accumbens medium spiny neurons target non-dopaminergic neurons in the ventral tegmental area. *Journal of Neuroscience*, 31(21), 7811–7816.

---

## The Rescorla-Wagner (RW) Model

$$\Delta V_i = \alpha_i \beta \left(\lambda - \sum_j V_j\right)$$

- **Vᵢ** — associative strength of stimulus i (learned value)
- **αᵢ** — salience of stimulus i (cue-specific learning rate)
- **β** — learning rate for the outcome
- **λ** — maximum conditioning supported by the outcome
- **(λ − ΣVⱼ)** — prediction error: how surprising was the outcome?

#### Relation to SARSA

| RW Model | SARSA / TD |
|----------|------------|
| λ − ΣVⱼ | r + γ·Q(s₂,a₂) − Q(s₁,a₁) = δ |
| Associative strength V | Action-value Q(s, a) |
| Single timestep, no γ | Sequential across time with γ |

**Key advance of TD over RW:** TD assigns credit across time via γ; RW has no temporal depth

<p class="cite">Rescorla & Wagner (1972)</p>
