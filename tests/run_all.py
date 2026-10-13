"""Runs every test_* function of the test modules (no pytest needed): python tests/run_all.py [--rl]."""
import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, str(ROOT))
MODULES = ["test_spec_wrapper", "test_certificates", "test_validation", "test_convergence_frontier", "test_wear_evaluate", "test_regularizers", "test_paper_results", "test_report_assumptions"]
if "--rl" in sys.argv:
    MODULES.append("test_sb3")
failed = 0; passed = 0
for name in MODULES:
    mod = importlib.import_module(name)
    for fn in [getattr(mod, k) for k in dir(mod) if k.startswith("test_")]:
        try:
            fn(); passed += 1; print(f"ok   {name}.{fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {name}.{fn.__name__}"); traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
