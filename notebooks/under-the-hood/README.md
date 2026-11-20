# Technical tutorials

These notebooks connect the paper's results to executable examples. The wear tutorials use a grid-based benchmark for a continuous simulator. The regularization tutorial also examines an analytically tractable policy model. The four [applied notebooks](../../README.md#tutorials-and-examples) describe the workflow on fuel, cash, battery and navigation tasks.

| Notebook | Calculation or comparison | Dependencies |
|---|---|---|
| [T1. Fixed failure prices and the wear benchmark](T1_price_and_exact_frontier.ipynb) | Clipped-value decomposition, a grid-based policy comparison, and validation under geometric maintenance access | Core and Matplotlib |
| [T2. PPO and adaptive pricing on the wear problem](T2_training_against_an_exact_benchmark.ipynb) | Trained policies at several prices, adaptive failure pricing, and a sensitivity test at a reduced wear allowance | RL and notebook extras |
| [T3. Action and parameter regularization](T3_roles_of_the_regularizers.ipynb) | Noise and drift in the population model, prior-policy evaluation, and measured effects of finite-network penalties in PPO | RL and notebook extras |
| [T4. Statistical validation at fixed and geometric horizons](T4_validation_in_detail.ipynb) | Exact binomial endpoints, error allocation, and a value certificate for a fixed physical horizon | Core and Matplotlib |

The theory guide provides the conditions behind these calculations. For an exact solution of a known finite transition model, see the dynamic programming guide. [Theory and implementation](../../docs/under-the-hood/decisions-and-theory.md) · [Results and functions](../../docs/under-the-hood/results-map.md) · [Exact dynamic programming](../../docs/dynamic-programming.md)
