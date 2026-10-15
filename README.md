# bpg: clipped barrier policy gradients under resource budgets

**Keep the reinforcement learner you already use. Add a budget. Know what the trained policy delivers, and prove it.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab)
![License: MIT](https://img.shields.io/badge/license-MIT-2e9e5b)
![Core: numpy + scipy](https://img.shields.io/badge/core-numpy%20%2B%20scipy-8a8a8a)

`bpg` is the companion toolbox of the paper *The Price of Failure Risk: Clipped Barrier Policy Gradients with
Convex Regularization* (Malo, Viitasaari, Suominen, Vilkkumaa, Tahvonen). It is for practitioners who train a
policy under a cumulative resource budget (a wear allowance, a fuel tank, an emissions cap, a bycatch quota)
and must say, at the end, how often the deployed policy exhausts the budget and in what sense it solves the problem.

![What the toolbox is about](docs/images/overview.png)

## The idea in one paragraph

Carry the remaining budget in the state, subtract a reserve barrier from the reward, and train a standard learner
such as PPO or SAC unchanged. That ease of use is why the method is worth understanding. An unbounded barrier gives
every exploratory policy the value $-\infty$, so the combined reward is clipped from below at $-m$. The paper shows
that $m$ is then a **price of failure risk**: any optimizer of the clipped objective is optimal at the risk it attains,
its risk exceeds any other policy's by at most order $1/m$, and its risk can be certified from its value or from
validation data. It also explains what the regularizers do: action regularization holds the policy near a trusted
reference, and parameter regularization makes the learning problem well posed and holds it near a prior policy.
The toolbox turns these results into code.

## What the toolbox does

| Step | You get | Main functions |
|---|---|---|
| **Configure** | A requirement stated before any mechanism; a conservative starting price; a validation budget; the conditions under which another method is better; regularizer advice with reasons | `BudgetSpec`, `Requirement`, `plan`, `advise_regularizers` |
| **Train** | A drop-in wrapper for any Gymnasium environment, and callbacks for Stable-Baselines3: an adaptive multiplier, an action anchor, a parameter anchor, parameter noise, a training monitor | `BudgetWrapper`, `DualAscent`, `ActionAnchor`, `ParameterAnchor`, `ParameterNoise`, `BudgetMonitor` |
| **Supply evidence** | Evaluation of the deployed candidate; exact binomial validation with the error shared across candidates; every certificate of the paper with its hypotheses; checks across prices; trace diagnostics | `evaluate`, `validation`, `certificates`, `frontier`, `convergence` |
| **Interpret** | A report that says in what sense the candidate solves the problem, what each regularizer did to it, and what the evidence does **not** establish | `interpret`, `regularizer_effects` |

Every bound the toolbox returns carries its value, the paper result it comes from, the hypotheses it needs, and a
flag when it is uninformative.

## Quick start

```bash
git clone <this repository> bpg-toolbox && cd bpg-toolbox
python -m venv ~/.venvs/bpg-toolbox && source ~/.venvs/bpg-toolbox/bin/activate
pip install -e ".[all]"      # core + Gymnasium, Stable-Baselines3, PyTorch, Jupyter, pytest
python tests/run_all.py --rl # 43 checks
```

The core needs only `numpy` and `scipy` (`pip install -e .`). The wrapper uses Gymnasium when it is installed, and
the callbacks need Stable-Baselines3 (`pip install -e ".[rl]"`).

### Sixty seconds: certify a candidate and read the report

```python
from bpg import Requirement, Stage, evaluate, interpret
from bpg.examples.wear import WearModel, make_wear_env, IncumbentRule

model = WearModel()                                   # an asset run under a wear allowance (the paper's running example)
spec = model.spec(ubar=1.0)                           # budget b = 1, utility discount 0.95, utility in [0, 1]
req = Requirement(T=19, alpha=0.05, zeta=0.05)        # breakdown within 20 periods: at most 5%, at 95% confidence
stage = Stage(m=1.0)                                  # the price of failure risk

policy = IncumbentRule(span=3.0, std=0.1, seed=0)     # any callable obs -> action: a rule or a trained network
ev = evaluate(lambda: make_wear_env(stage, model), policy, 10_000, spec=spec, stage=stage, T=19)
print(interpret(spec, req, stage, ev).text())
```

```text
Candidate under price m = 1; requirement p_19 <= 0.05 at error 0.05.

Measured on fresh episodes:
  n=10000: p_T censored by the geometric clock (105 failures counted for q_beta), q_beta=0.0105, U=2.729±0.018, ...

Verdict: CERTIFIED by direct validation at the specified horizon.

In what sense the candidate is a solution:
  - Optimization error is unknown: no optimality or convergence claim is made for this candidate. ...

Evidence and bounds:
  - direct validation under the geometric clock: 105 failures in 10000 episodes; exact binomial upper endpoint 0.01234 ...
  - finite-horizon requirement validated through q_beta <= alpha beta^T = 0.01887: passes (p_19 <= 0.03269) ...

Not established by this evidence:
  - eventual risk: a small q_beta or a certificate at T does not bound P(tau < inf) ...
  - other horizons: certification at T = 19 does not certify a longer horizon ...
  - other candidates: the mean-action policy, a policy with reduced dispersion, or a retrained one is a different candidate ...
  - convergence of this training run: ... the trace diagnostics are evidence about this run, not a convergence theorem
  - model error: every certificate and validation is conditional on the simulator
```

### Use it with your own environment and learner

```python
import gymnasium as gym
from stable_baselines3 import PPO
from bpg import BudgetSpec, Stage, BudgetWrapper
from bpg.callbacks import BudgetMonitor

def fuel(obs, action, next_obs, reward, info):          # the resource each step consumes (nonnegative)
    main = (min(max(action[0], 0.0), 1.0) + 1.0) / 2 if action[0] > 0 else 0.0
    side = min(max(abs(action[1]), 0.5), 1.0) if abs(action[1]) > 0.5 else 0.0
    return [0.30 * main + 0.03 * side]

spec = BudgetSpec(budgets=[22.0], beta=0.999, ubar=100.0, u_minus=100.0, horizon=1000)   # a 22-unit fuel tank
stage = Stage(m=0.5, clip_utility=False)   # the crash penalty lies below -m, so price the failure only
env = BudgetWrapper(gym.make("LunarLanderContinuous-v3"), cost=fuel, spec=spec, stage=stage)
model = PPO("MlpPolicy", env, gamma=spec.beta).learn(1_000_000, callback=BudgetMonitor(spec.beta, stage.m, horizon=spec.horizon))
```

The wrapper appends the remaining budget to the observation, replaces the reward by the clipped barrier reward,
ends the episode at failure with the discounted total of the price, and records what the certificates need. Options
cover an interior barrier (`Stage(theta=...)`), a flat penalty updated by dual ascent (`Stage(penalty="multiplier")`),
a geometric clock that implements the discount exactly (`geometric_stop=True`), a misjudged allowance
(`true_budget_scale=0.9`), and resources whose step use can be negative (`RunningMaxCost`).

## Tutorials

Executed notebooks with their outputs, in [`notebooks/`](notebooks/) (times on one CPU core):

| Notebook | What it shows | Needs | Time |
|---|---|---|---|
| [1. Quickstart](notebooks/01_quickstart.ipynb) | The clipped reward and the identity behind every result; a plan; candidates against the exact frontier; validation; the report | numpy | 20 s |
| [2. Training with PPO](notebooks/02_training_with_ppo.ipynb) | Training through the wrapper; stochastic versus mean-action candidates; training at several prices and the optimization error; an adaptive multiplier; the margin bought by the barrier | RL extras | 11 min |
| [3. Regularizers and their roles](notebooks/03_regularizers_and_roles.ipynb) | Action versus parameter regularization; noise, pull and spread on the solvable flow; convergence versus bias; the prior is a policy; parameter noise on PPO; joint use | RL extras | 12 min |
| [4. Validation and reporting](notebooks/04_validation_and_reporting.ipynb) | Resolution floors; why the error is allocated before selecting; the geometric clock versus a fixed horizon; when value certificates are informative; a lump sanction with your own vectorized evaluator | numpy | 1 min |
| [5. Bring your own problem](notebooks/05_bring_your_own_problem.ipynb) | A new task written from scratch, with no exact benchmark: structured policies first, pilot runs at several prices, what can and cannot be claimed | RL extras | 13 min |

The scripts in [`examples/`](examples/) cover the same ground from the command line.

## When to use another method

The bounded barrier is a reasonable first method when failures can be permitted in simulation and deployment has a
positive tolerance or a finite failure cost. The paper quantifies this choice; it does not claim that the method
outperforms methods built for particular constraint structures. Another approach is the better choice when

1. **no failure is permissible and a controlled-invariant set is known** (an almost-sure method: an infinite failure
   penalty, a control-barrier safety layer, or a method that learns which actions keep probability-one constraints);
2. **a prescribed level is attainable with slack and the multiplier's step can be tuned** (an adaptive multiplier, as
   in a Lagrangian chance-constrained policy gradient);
3. **violations during training matter** (constrained policy optimization);
4. **the loss grows with the amount by which the consumption exceeds the budget** (a CVaR constraint on the
   cumulative consumption);
5. **the model is known and small** (an exact dynamic program; the toolbox's certificates and validation apply to its
   policy too).

Two further cases concern the barrier itself: a required price that charges far more for a failure than an episode
earns (reward normalization or a prescribed-level formulation), and an allowance error of known size (tighten the
allowance instead of adding an interior margin). `plan` states these conditions for your problem before anything is
trained.

## What the evidence means

The report separates three kinds of statements, as the paper does.

* **The reading of the objective.** With a known optimization error $\delta$, a fixed-price candidate is optimal at
  the risk it attains, up to $\delta$. With an unknown error the report makes no optimality claim.
* **Certificates and validation.** Value certificates need no optimization guarantee but degrade with the horizon.
  The exact binomial test of the deployed candidate at its horizon, with the error shared across every candidate
  inspected, is the deployment decision. Zero failures certify the resolution floor, never zero.
* **Convergence.** The population flow converges exponentially above a curvature threshold that grows with the price;
  near the threshold the minimizer lies close to the prior, so strengths are set for conditioning. Under a potential
  regularizer, continuous-time training of a finite network is that flow. Discrete stochastic training with PPO or
  SAC is outside both results, and trace diagnostics are evidence about one run.

Not established by any of this: eventual risk from a small geometric risk, other horizons, other candidates (the
mean-action policy is a different candidate), and behavior under a different simulator.

## Documentation

* [Practitioner's guide](docs/practitioner-guide.md): one section per decision, with the functions that implement it.
* [Worked examples](docs/worked-examples.md): the paper's wear-allowance, fuel-tank, roles, emissions and benchmark-suite studies as a practitioner would run them.
* [Results map](docs/results-map.md): every paper result and the function that implements it.
* [Supplementary results](docs/supplementary-results.md): two certificates the toolbox adds, with proofs.
* [Development](docs/development.md): the local environment, VS Code, tests and rebuilding the notebooks.

## Citing results of the paper

The toolbox cites results by the titles the paper gives them, never by statement or section numbers, so a new
revision of the paper does not change the documentation. All citations go through one registry,
[`bpg/paper.py`](bpg/paper.py): `cite("fixed_price", "ii")` returns
`paper, 'Fixed price of failure risk' (price-risk law)`, and `bpg.PAPER["checked_against"]` records the revision
against which the toolbox was last checked.

## Citing

If you use the toolbox, please cite the paper and the software; [`CITATION.cff`](CITATION.cff) has the details.

## License

MIT; see [`LICENSE`](LICENSE).
