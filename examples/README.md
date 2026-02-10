# Examples

Place standalone runnable helpers and scripts in this directory. Execute them
directly from the project root, e.g. `python examples/<script>.py`, when you
need procedural utilities. The SARSA walkthrough ships as
`examples/sarsa.ipynb`; launch it via `jupyter lab examples/sarsa.ipynb`.
Keep imports relative to the installed package (`from sarsa import sarsa`) so the
examples remain portable.

## Contents

| File | Description |
|------|-------------|
| `sarsa.ipynb` | Primary walkthrough notebook demonstrating the full SARSA fitting workflow. |
| `experiment.py` | Task-specific helpers: state construction, onset encoding, and data processing. |
| `M1.csv` | Sample behavioural dataset (~6.3 MB) used by the notebook. |

## Prerequisites

Install the optional tooling before opening the notebook:

```bash
uv sync --extra examples       # recommended
# or
pip install -e .[examples]
```
