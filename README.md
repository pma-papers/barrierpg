# Reinforcement learning under resource budgets

`barrierpg` supports several formulations for resource-budget problems. A retailer may face a known cost of insolvency, a lander may have a maximum tolerated probability of losing the vehicle, and a robot may be subject to a limit on mean energy use. These specifications lead to different training objectives, constraints and validation measurements.

| Method or formulation | Support in the toolbox |
|---|---|
| Exact dynamic programming | Bellman backward induction for a known finite model with stochastic consumption and an integer budget. Solve a priced or pathwise feasible policy and compute its model-based failure probability. |
| Fixed failure price and reserve penalties | A Gymnasium reward wrapper for compatible learners such as proximal policy optimization (PPO). Use a failure-only price, a fixed lump penalty, or an optional penalty for small reserves. |
| Expected-cost constraint | A single-cost constrained policy optimization (CPO) actor update on the original utility and a separate cost signal. An optional OmniSafe adapter supports full CPO training. |
| Chance constraint at a specified horizon | Exact one-sided binomial validation of first crossing. An adaptive failure multiplier or a CPO update with the corresponding event cost can be used in training. Finite-scenario selection is also available. |
| Conditional value-at-risk (CVaR) constraint | Upper-tail risk and policy selection on finite scenarios, plus a barrier-free trust-region actor update using a Gaussian CVaR surrogate. |
| Pathwise feasibility | Backward induction in a finite transition model, or selection from candidate policies that meet the cost limit in every positive-probability scenario. |

The formulation specifies the objective, the quantity constrained and the horizon. The learning algorithm produces a candidate policy. Validation then evaluates that policy against the deployment requirement. CPO and CVaR can both be used with unmodified utility. Action penalties, parameter penalties and an interior barrier are optional objective terms. [Formulation tutorial](docs/formulation-comparison.md)

The budget wrapper and risk certificates follow *The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization* (Malo, Viitasaari, Suominen, Vilkkumaa and Tahvonen). For a first-crossing requirement, the policy observes the remaining reserve. Failure occurs at the first action after which any reserve is negative. A reserve of exactly zero is permitted.

## First-crossing risk and the priced objective

A finite-horizon chance constraint takes the form $p_T(\pi) = \Pr_\pi(\tau \le T) \le \alpha$. Here $\tau$ is the first failure index, $T$ is the last action index covered by the requirement, and $\alpha$ is the tolerated failure probability. The confidence level for validation is $1-\zeta$. For example, a policy can be required to keep its annual insolvency probability at or below 2%, with 95% confidence from simulated years.

One way to train a candidate is to use a fixed failure price. In the continuing formulation, the clipped value equals stopped utility less the interior loss and a charge of $m/(1-\beta)$ per unit of geometric failure risk. Stopped utility is the discounted utility earned before failure. Geometric risk weights a failure at action $t$ by $\beta^t$, where $\beta$ is the utility discount. An optional interior barrier deducts utility as the reserve approaches zero. With a physical horizon, the per-period charge ends at the last action of the task. A known one-off loss can be charged at failure through a fixed lump penalty.

The attained risk depends on the policy class and the policy returned by training. A price therefore provides a training incentive, and independent validation establishes whether a fixed candidate meets the deployment tolerance. When several candidates or horizons are assessed on the same validation data, the statistical error is allocated across those assessments.

## Installation

```bash
git clone https://github.com/pma-papers/barrierpg.git
cd barrierpg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rl]"
```

Python 3.10 or later is required. Install `-e .` for the NumPy and SciPy core, `-e ".[rl,problems]"` for PPO and the Box2D or MuJoCo examples, or `-e ".[all]"` to include the Jupyter and test dependencies. Training on the navigation task uses a separate Safety-Gymnasium environment. [Development and installation notes](docs/development.md)

## Solve a known finite model

When the transition probabilities and consumption outcomes are known, use `solve_budget_dp` to compute an optimal decision at each state, time and reserve. The solver supports a fixed failure price, a lump charge and pathwise feasibility, with optional interior loss. It reports the selected policy's stopped utility and failure probability under the supplied model.

```bash
python examples/solve_finite_budget.py
```

The example compares two policies under stochastic wear. [Dynamic programming guide](docs/dynamic-programming.md)

## Train with CPO or CVaR

Supply the original utility to the actor update and define the constrained cost separately. The dispatch example uses the same actions and outcomes for a CPO mean-cost constraint and a Gaussian CVaR constraint. It evaluates the final policy's mean cost and exact tail cost under the model.

```bash
python examples/train_risk_constrained.py
```

The example requires the RL extra. [CPO and CVaR without an interior barrier](docs/formulation-comparison.md#cpo-and-cvar-training-without-an-interior-barrier)

## Validate a fixed policy

The retailer starts each 52-week episode with 2,500 units of cash. We evaluate its base-stock ordering rule on 2,000 simulated years. An episode fails when its cash balance first becomes negative. The stage charges 5,000 units once at insolvency. The policy is fixed before these episodes, and the test covers the full year.

```python
from barrierpg import Requirement, Stage, evaluate, interpret
from barrierpg.problems.inventory import BaseStockRule, Retailer, make_inventory_env

retailer = Retailer()
spec = retailer.spec()
requirement = Requirement(T=51, alpha=0.02, zeta=0.05)
stage = Stage(m=0.0, penalty="multiplier", multiplier=5_000.0)
rule = BaseStockRule(retailer, cover=2.0, reserve=300.0)

episodes = evaluate(
    lambda: make_inventory_env(stage, retailer),
    rule.deterministic,
    2_000,
    spec=spec,
    stage=stage,
    T=requirement.T,
    seed=0,
)
report = interpret(spec, requirement, stage, episodes, candidate_name="the base-stock rule")
print(report.summary(resource="cash", period="weeks"))
```

In this run the rule failed in 6 of 2,000 episodes. The observed frequency was 0.30%, and the exact one-sided 95% upper confidence bound was 0.652%. The rule met the 2% tolerance in the specified simulator. The same procedure applies to a trained policy, including its action sampling and any observation normalization used in deployment. `report.text()` gives the assumptions behind the available bounds.

## Train on a budget-constrained task

To use an existing Gymnasium task, provide a function that returns one nonnegative consumption per resource after each action. In `BudgetSpec`, specify the budgets, utility discount, and physical horizon where applicable. `BudgetWrapper` adds reserve information to the observation and accounts for the failure charge at termination. For a task with a fixed horizon, include time in the observation when the remaining time affects decisions.

The following example trains on the paper's landing problem. The two failure events are fuel exhaustion and loss of the lander. `landing.stage(1.0)` sets the failure charge at the reference action to one landing revenue. The training aid is disabled during evaluation. The short run illustrates training and subsequent evaluation. Its output includes the validation decision for the fitted candidate.

```python
from stable_baselines3 import PPO

from barrierpg import Requirement, evaluate, interpret, sb3_policy
from barrierpg.problems.landing import Landing, make_landing_env

landing = Landing()
spec = landing.spec()
requirement = Requirement(T=spec.horizon - 1, alpha=0.05, zeta=0.05)
stage = landing.stage(1.0)
env = make_landing_env(stage, landing)
try:
    model = PPO("MlpPolicy", env, gamma=spec.beta, seed=0)
    model.learn(32_768)
finally:
    env.close()

episodes = evaluate(
    lambda: make_landing_env(stage, landing, aid=False),
    sb3_policy(model),
    500,
    spec=spec,
    stage=stage,
    T=requirement.T,
    seed=100_000,
)
print(interpret(spec, requirement, stage, episodes).text())
```

The practitioner guide covers price selection, reserve penalties, action and parameter regularization, and validation design. The template defines a cost function and wraps an existing simulator. [Practitioner guide](docs/practitioner-guide.md) · [Custom task template](examples/your_problem.py)

## Tutorials and examples

| Applied notebook | What it examines |
|---|---|
| [Lander with a fuel tank](notebooks/01_lander_fuel_tank.ipynb) | Fine-tuning an existing controller under a fuel budget and comparing action and parameter references. This notebook uses the earlier fuel-tank task and its archived runs. |
| [Retailer with a cash balance](notebooks/02_retailer_cash.ipynb) | Modeling replenished cash, charging an insolvency loss once, and comparing learned policies with an incumbent ordering rule. |
| [Robot with a battery](notebooks/03_robot_battery.ipynb) | Including remaining time in the observation and assessing reserve under a 10% reduction in available energy. |
| [Navigation with a damage allowance](notebooks/04_navigation_allowance.ipynb) | Distinguishing a mean damage limit from a probability of exceeding the allowance and validating the latter at a fixed horizon. |

The notebooks pair short runnable examples with results from longer training runs. The run configuration, training traces, trained policies and episode-level measurements are available in the repository. [Training runs and reproduction commands](campaigns/README.md)

- For the choice among a fixed failure price, an expected-cost limit, a chance constraint, a conditional value-at-risk (CVaR) limit, and pathwise feasibility, see the [formulation tutorial](docs/formulation-comparison.md). Its [dispatch example](examples/train_risk_constrained.py) runs barrier-free CPO and Gaussian CVaR updates on the same utility and separate cost outcomes.
- For empirical comparisons on the wear-allowance task, see the [paired benchmark](docs/wear-benchmark.md) and the [guide to residual policies](docs/residual-networks.md).
- For the clipped-value identity, the wear benchmark, the roles of regularization, and validation at geometric and fixed horizons, see the [technical notebooks](notebooks/under-the-hood/), the [theory guide](docs/under-the-hood/decisions-and-theory.md) and the [map from results to functions](docs/under-the-hood/results-map.md). The paper's experiments can be reproduced through the [replication repository](https://github.com/pma-papers/bpg-replication).

## Citation and license

The toolbox is distributed under the [MIT license](LICENSE). Citation metadata and archive information are in [`CITATION.cff`](CITATION.cff) and the [Zenodo archive](https://doi.org/10.5281/zenodo.23597362).
