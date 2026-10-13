"""Build the tutorial notebooks from their sources and execute them.

Sources live in tools/notebook_sources/*.py in the "percent" format: a line '# %%' starts a code cell and
'# %% [markdown]' a markdown cell whose lines are written as comments. The builder writes notebooks/<name>.ipynb
and executes it in place, so the outputs show on GitHub.

    python tools/build_notebooks.py                 # build and execute all
    python tools/build_notebooks.py 01 03           # only the notebooks whose names start with these prefixes
    python tools/build_notebooks.py --no-execute    # build without executing

Run it with the Python of the toolbox environment (it needs nbformat, nbclient and ipykernel, and the RL extras
for notebooks 02, 03 and 05).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "tools" / "notebook_sources"
OUT = ROOT / "notebooks"


def parse(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            lines = []
            for line in body.split("\n"):
                if line.startswith("# "):
                    lines.append(line[2:])
                elif line.startswith("#"):
                    lines.append(line[1:])
                else:
                    lines.append(line)
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.split("\n"):
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def build(src: Path, execute: bool = True) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(src.read_text())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    out = OUT / (src.stem + ".ipynb")
    if execute:
        t = time.time()
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
        print(f"executed {out.name} in {time.time() - t:.0f}s")
    nbformat.write(nb, out)
    return out


def main(argv):
    execute = "--no-execute" not in argv
    prefixes = [a for a in argv if not a.startswith("--")]
    OUT.mkdir(exist_ok=True)
    for src in sorted(SOURCES.glob("*.py")):
        if prefixes and not any(src.stem.startswith(p) for p in prefixes):
            continue
        build(src, execute)


if __name__ == "__main__":
    main(sys.argv[1:])
