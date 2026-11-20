# Reproduce the tutorial training runs

The tutorials combine short live runs with saved outcomes from longer PPO training. This directory contains the task specifications, commands, fitted policies and evaluation episodes. Each run is identified by problem, configuration and training seed.

| File | Content |
|---|---|
| `problems.py` | the four problems (environment, requirement horizon, PPO settings) and their configurations |
| `run.py` | trains and evaluates one (problem, configuration, seed) and writes its record |
| `launch.sh`, `run_one.sh` | run a list of jobs in parallel; `jobs-*.txt` are the lists that produced the results |
| `records.py` | reads the records for the notebooks: tables over seeds, evaluations, trained policies |
| `results/<problem>/<configuration>/seed<k>/` | `model.zip` (the trained policy), `vecnormalize.pkl` (observation statistics, when used), `record.json` |

A configuration fixes the training objective, number of interactions, regularizers and initialization. The stage uses a per-period price or a lump penalty, and some configurations adapt the lump penalty during training. Initial policies include random starts, actors fitted to a rule and checkpoints from another configuration.

`record.json` holds the configuration, the PPO settings (the anchored configurations were trained with the coupled
form of the parameter anchor, the drift added to the gradient before Adam's step, which `Config.anchor_decoupled=False`
pins; records written from 0.6.1 on say so under `regularizers.anchor_form`), the training traces per rollout (failure rate, utility,
dispersion, and the multiplier when it adapts), and the evaluation of the trained stochastic policy on fresh
episodes, one entry per episode: whether and when it failed, its utility, and the problem's own measures (landed or
crashed, fill rates and cash, distance and energy, damage). Runs that start from another policy also record the
evaluation of that starting policy, and some configurations add evaluations under a different budget (a battery
that holds 10% less than the robot believes). `records.Campaign(...).evaluation(...)` rebuilds a `barrierpg.Evaluation`
from these entries without simulating again.

## Running

```bash
python campaigns/run.py retailer lump-5000 --seed 4                     # one run, written to campaigns/results/
python campaigns/run.py robot price-1 --seed 1 --steps 200000 --eval 50 --out /tmp/pilot   # a short pilot elsewhere
sh campaigns/launch.sh campaigns/jobs-retailer.txt 6                    # a list of runs, six at a time
```

The navigation problem needs the Python environment with Safety-Gymnasium; `run_one.sh` uses `BPG_SAFETY_PYTHON` for
it (default `~/.venvs/constrained-rl-safety310/bin/python`) and `BPG_PYTHON` for the others. Runs that start from
another configuration's policy (the lander's fine-tuning starts from `controller`; `adaptive-1pct-warm` from
`lump-20000`) need that run of the same seed first.

Times on one core of a recent laptop, with the machine shared by a dozen runs: about 6 minutes per retailer run,
5 minutes per robot run, 10 minutes per lander controller and a few minutes per lander fine-tuning, and about
35 minutes per navigation run.

The configurations trained without a price are named `no-price`, `tank-no-price` and `no-price-no-clock`. Until
2026-11-12 they were named `unpriced`, `tank-unpriced` and `unpriced-no-clock`; `Campaign` and `run.py` still accept
those names with a `DeprecationWarning`, and the `record.json` of each of these runs keeps the name under which it
was run.
