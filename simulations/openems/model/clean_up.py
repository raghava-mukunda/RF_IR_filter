"""
Clean all V11 generated simulation/result/plot files.
"""

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATHS = [
    PROJECT_ROOT / "simulations" / "openems" / "results" / "chebyshev_v11",
    PROJECT_ROOT / "results" / "em" / "chebyshev_v11",
    PROJECT_ROOT / "results" / "plots" / "chebyshev_v11",
]

for path in PATHS:
    if path.exists():
        print(f"Removing: {path}")
        shutil.rmtree(path)
    else:
        print(f"Not present: {path}")

print("\nV11 cleanup complete.")
