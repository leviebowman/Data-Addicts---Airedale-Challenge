"""Run every Analysis/ script in order.  Takes about 2 minutes.

Run:  .venv\\Scripts\\python.exe Analysis\\run_all.py
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SCRIPTS = ["01_data_audit.py", "02_inference.py", "03_classification.py", "04_cost_model.py",
           "05_segmentation.py", "06_portfolio_extensions.py"]

for s in SCRIPTS:
    t = time.time()
    print(f"\n{'=' * 80}\n{s}\n{'=' * 80}")
    runpy.run_path(str(HERE / s), run_name="__main__")
    print(f"[{s} finished in {time.time() - t:.0f}s]")
