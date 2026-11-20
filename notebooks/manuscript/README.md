# Worked examples from the manuscript

These notebooks use the experiments reported in revision 78 of *The Price of Failure Risk*. Each states the optimization problem, shows the relevant model or update in code, and explains the results using the manuscript's figures. The saved outputs come from the recorded studies. The notebooks can be run without repeating the training campaigns.

| Study | What the notebook examines | Training or computation |
|---|---|---|
| [An asset operated under a wear allowance](M1_wear_allowance.ipynb) | Set a failure price or train with an adaptive multiplier, compare with a grid benchmark, validate geometric risk, and test a smaller allowance | 8 million interactions per policy, 8 seeds per configuration, 20,000 evaluation episodes per policy |
| [A learned lander paid for each landing](M2_paid_landing.ipynb) | Define a physical fuel budget and a loss event, validate failure by the mission horizon, and compare policies at a common failure charge | 30 million interactions for the starting controller, then 60 million per charge, 10 seeds per charge, 2,000 evaluation episodes per policy |
| [Regularization in the analytical gradient-flow example](M3_analytical_gradient_flow.ipynb) | Relate the parameter law to the action density, compare population objective decay, and read fixed-strength and scheduled particle runs | Finite-volume population calculations and, for the schedule comparison, 8 runs of 10,000 particles |

Install the toolbox and notebook extra with `pip install -e '.[notebooks]'`. For wear and landing, install the RL extra with `pip install -e '.[notebooks,rl]'`. The landing simulator also requires `pip install 'gymnasium[box2d]'`. The [technical notebooks](../under-the-hood/) provide smaller executable demonstrations of the implementations.

## Data and figures

The [replication repository](https://github.com/pma-papers/bpg-replication/tree/174651d8ffe7ba5a521fb06e109d8387e58c0cf9) holds the protocols, run records, and plotting code. The compact tables in [data/](data/) retain the study values used in these notebooks.

- Wear summarizes the [recorded campaign](https://github.com/pma-papers/bpg-replication/blob/174651d8ffe7ba5a521fb06e109d8387e58c0cf9/wear-allowance/wear-summary-v41.json) and recomputes the paper's binomial pass counts. The frontier, smaller-allowance figure and training traces come from the manuscript and Online Resource 1.
- Landing uses the [reported summary](https://github.com/pma-papers/bpg-replication/blob/174651d8ffe7ba5a521fb06e109d8387e58c0cf9/fuel-tank/lunar-landing-paper-numbers.json) and the failure counts, stopped utility and priced risk of the 80 policies in the [main campaign](https://github.com/pma-papers/bpg-replication/tree/174651d8ffe7ba5a521fb06e109d8387e58c0cf9/fuel-tank/lunar/ledger-staged-anneal2). The notebook recomputes the own-charge comparison. Its 2% tolerance is an example of applying a deployment requirement to those counts.
- The analytical example uses the manuscript's setup, flow and schedule figures. The [solvable-flow study](https://github.com/pma-papers/bpg-replication/tree/174651d8ffe7ba5a521fb06e109d8387e58c0cf9/solvable-flow) gives the computation scripts and numerical records.

The notebook figures are PNG renders of the revision-78 figure PDFs, with the plotted data, labels and layout preserved. Credit Pekka Malo, Lauri Viitasaari, Antti Suominen, Eeva Vilkkumaa and Olli Tahvonen. The study figures and data are distributed under [CC BY 4.0](https://github.com/pma-papers/bpg-replication/blob/174651d8ffe7ba5a521fb06e109d8387e58c0cf9/LICENSE-records.md). The notebook code follows the [toolbox licence](../../LICENSE).
