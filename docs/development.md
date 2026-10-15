# Development

## The environment

The toolbox uses one virtual environment, kept outside the repository at `~/.venvs/bpg-toolbox` (the repository
sits in a synced folder, and an environment inside it makes the editor and the sync client crawl its 100,000 files).

```bash
python3 -m venv ~/.venvs/bpg-toolbox
source ~/.venvs/bpg-toolbox/bin/activate
pip install -e ".[all]"     # the package in editable mode, with Gymnasium, Stable-Baselines3, PyTorch, Jupyter and pytest
```

The exact package set in use is frozen in `../environments/bpg-toolbox.txt`.

Extras: `rl` (Gymnasium, Stable-Baselines3, PyTorch), `notebooks` (Jupyter, matplotlib, nbformat, nbclient),
`test` (pytest), and `all`. The core (`pip install -e .`) needs only numpy and scipy.

The paper's fuel-tank test needs Box2D (`pip install "gymnasium[box2d]"`). Building `box2d-py` needs a `swig`
executable on the path; on macOS, `brew install swig` provides it. Without Box2D that test is skipped.

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

The checks reproduce numbers of the paper where the toolbox can compute them: the curvature deficit of the solvable
flow ($C_m=1940$, certified $\kappa=2037$), the plateau value $-2.2533$, the stationary spreads of the roles study, the
protocol's noise strengths, the incumbent rule's risk and utility in the wear example, the price-sequence reading of
the emissions study, and a step-by-step reproduction of the fuel-tank environment.

## Notebooks and figures

The tutorial notebooks are generated from plain-text sources in `tools/notebook_sources/` (one `.py` file per
notebook in the "percent" format: `# %%` starts a code cell, `# %% [markdown]` a markdown cell). Edit the source, then
rebuild and execute:

```bash
python tools/build_notebooks.py          # all notebooks (about 40 minutes on one core; notebooks 2, 3 and 5 train PPO)
python tools/build_notebooks.py 01 04    # selected notebooks
python tools/show_outputs.py notebooks/01_quickstart.ipynb   # print the text outputs for review
python tools/make_readme_figures.py      # the figure on the landing page
```

## Following a new revision of the paper

The toolbox cites results by the titles the paper gives them and never by statement, section, table or equation
numbers, so a renumbering needs no change. A new revision is checked for content instead.

1. Check that every title in `RESULTS` of `bpg/paper.py` still names a statement of the paper; rename a key's title
   if the paper renamed the result.
2. Check what changed in substance (statements, the cases of *Which formulation to prefer*, the studies' tables) and
   update the functions, `docs/` and the notebook sources that use it.
3. Set `PAPER["checked_against"]` to the revision checked.
4. Run the tests (`test_no_statement_numbers_in_printed_text` fails on any number in printed citations or reports),
   search the documentation for numbers with
   `grep -rnE "(Proposition|Theorem|Corollary|Remark|Lemma|Table|Section|Appendix) [A-Z]?\.?[0-9]" bpg docs examples tools README.md`,
   and rebuild the notebooks.
5. Record the changes in `CHANGELOG.md`.
