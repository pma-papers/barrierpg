# Campaigns of the tutorial problems

The notebooks of the toolbox load trained policies instead of training them, so that they run in minutes. This
folder holds what produced those policies: the definitions, the scripts, and one record per run.

| File | Content |
|---|---|
| `problems.py` | the four problems (environment, requirement horizon, PPO settings) and their configurations |
| `run.py` | trains and evaluates one (problem, configuration, seed) and writes its record |
| `launch.sh`, `run_one.sh` | run a list of jobs in parallel; `jobs-*.txt` are the lists that produced the results |
| `records.py` | reads the records for the notebooks: tables over seeds, evaluations, trained policies |
| `results/<problem>/<configuration>/seed<k>/` | `model.zip` (the trained policy), `vecnormalize.pkl` (observation statistics, when used), `record.json` |

A configuration fixes everything a run needs: the stage (a price, a lump sum charged once, or an adaptive multiplier
toward a tolerance), the number of training steps, the regularizers, and where the policy starts (from scratch, from
a network fitted to an incumbent rule, or from the trained policy of another configuration).

`record.json` holds the configuration, the PPO settings (the anchored configurations were trained with the coupled
form of the parameter anchor, the pull added to the gradient before Adam's step, which `Config.anchor_decoupled=False`
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
