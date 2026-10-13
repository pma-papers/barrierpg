"""Print the text outputs of an executed notebook (for review): python tools/show_outputs.py notebooks/01_quickstart.ipynb"""
import sys
import nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            print(f"--- cell {i} ---\n{o.text.rstrip()}")
        elif o.output_type in ("execute_result", "display_data"):
            if "text/plain" in o.data and "image/png" not in o.data:
                print(f"--- cell {i} result ---\n{o.data['text/plain']}")
            elif "image/png" in o.data:
                print(f"--- cell {i}: figure ---")
        elif o.output_type == "error":
            print(f"--- cell {i} ERROR {o.ename}: {o.evalue}")
