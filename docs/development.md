# Development

## The environment

The toolbox uses one virtual environment, kept outside the repository at `~/.venvs/bpg-toolbox` (the repository
sits in a synced folder, and an environment inside it makes the editor and the sync client crawl its 100,000 files).

```bash
python3 -m venv ~/.venvs/bpg-toolbox
source ~/.venvs/bpg-toolbox/bin/activate
pip install -e ".[all]"     # editable, with Gymnasium (Box2D, MuJoCo), Stable-Baselines3, PyTorch, Jupyter and pytest
```

The exact package set in use is frozen in `../environments/bpg-toolbox.txt`.

Extras: `rl` (Gymnasium, Stable-Baselines3, PyTorch), `problems` (Box2D for the lander, MuJoCo for the robot),
`notebooks` (Jupyter, matplotlib, nbformat, nbclient), `test` (pytest), and `all`. The core (`pip install -e .`)
needs only numpy and scipy. The distribution and the package are both called `barrierpg`; before version 0.6.0 they
were called `bpg`, a name taken on PyPI by an unrelated package. On Python 3.14, Gymnasium's Box2D extra builds
`box2d-py`, which needs a `swig` executable on the path (on macOS, `brew install swig`); older Pythons get a prebuilt
wheel. Tests that need Box2D or MuJoCo are skipped without them.

The navigation problem needs Safety-Gymnasium 1.0, which requires Python 3.10 or older, Gymnasium 0.28 and MuJoCo 2.3,
so it lives in a second environment (here `~/.venvs/constrained-rl-safety310`) with the toolbox installed editable
and without dependencies (`pip install --no-deps -e .`). Only training and new evaluations of the navigation policies
need it; the navigation notebook reads the shipped records.

The step-by-step test of the paper's fuel-tank environment compares `BudgetWrapper` with the paper's own environment
class, which is shipped as a test fixture (`tests/fixtures/paper_fuel_lander.py`, copied verbatim from the replication
repository with the hash of its source file). It needs Box2D.

## VS Code

The repository carries workspace settings in `.vscode/settings.json` that point the Python and Jupyter extensions to
`~/.venvs/bpg-toolbox/bin/python`. They override a user-level `python.defaultInterpreterPath`. Open the repository folder itself
(File > Open Folder...), then

1. if the status bar does not show `bpg-toolbox`, run **Python: Select Interpreter** from the command palette and pick
   `~/.venvs/bpg-toolbox/bin/python`. VS Code remembers a choice made earlier for the folder, and that choice takes precedence over
   the workspace default; **Python: Clear Workspace Interpreter Setting** resets it;
2. in a notebook, use **Select Kernel > Python Environments > bpg-toolbox**;
3. the Testing panel runs `pytest` on `tests/`.

A new terminal in VS Code activates the environment automatically (`python.terminal.activateEnvironment`).

To register the environment as a named Jupyter kernel, usable from any Jupyter front end:

```bash
~/.venvs/bpg-toolbox/bin/python -m ipykernel install --user --name bpg-toolbox --display-name "Python (bpg-toolbox)"
```

## Tests

```bash
python tests/run_all.py          # numpy-only checks
python tests/run_all.py --rl     # also the Stable-Baselines3 and Gymnasium checks
python -m pytest tests -q        # the same checks under pytest
```

The checks cover the budget mechanics of the four tutorial problems, the validation and certificates, the plain
report, and they reproduce numbers of the paper where the toolbox can compute them: the curvature deficit of the
solvable flow ($C_m=1940$, certified $\kappa=2037$), the plateau value $-2.2533$, the stationary spreads of the roles
study, the incumbent rule's risk and utility in the wear example, and a step-by-step reproduction of the paper's fuel-tank environment.

## Campaigns

The notebooks of the four problems read trained policies from `campaigns/results/`; [`campaigns/README.md`](../campaigns/README.md)
explains how to rerun them. Rerunning a campaign changes the numbers in the notebooks' outputs, so rebuild those
notebooks afterwards.

## Notebooks and figures

The tutorial notebooks are generated from plain-text sources in `tools/notebook_sources/` (one `.py` file per
notebook in the "percent" format: `# %%` starts a code cell, `# %% [markdown]` a markdown cell). Edit the source, then
rebuild and execute:

```bash
python tools/build_notebooks.py          # all notebooks: the four problems (minutes each) and the under-the-hood track
python tools/build_notebooks.py 01 T4    # selected notebooks, by the start of their names
python tools/show_outputs.py notebooks/01_lander_fuel_tank.ipynb   # print the text outputs for review
python tools/make_readme_figures.py      # the figure on the landing page
```

## Following a new revision of the paper

The toolbox cites results by the titles the paper gives them and never by statement, section, table or equation
numbers, so a renumbering needs no change. A new revision is checked for content instead.

1. Check that every title in `RESULTS` of `barrierpg/paper.py` still names a statement of the paper; rename a key's
   title if the paper renamed the result.
2. Check what changed in substance (statements, the cases of *Which formulation to prefer*, the studies' tables) and
   update the functions, `docs/` and the notebook sources that use it.
3. Set `PAPER["checked_against"]` to the revision checked.
4. Run the tests (`test_no_statement_numbers_in_printed_text` fails on any number in printed citations or reports),
   search the documentation for numbers with
   `grep -rnE "(Proposition|Theorem|Corollary|Remark|Lemma|Table|Section|Appendix) [A-Z]?\.?[0-9]" barrierpg docs examples tools README.md`,
   and rebuild the notebooks.
5. Record the changes in `CHANGELOG.md`.
