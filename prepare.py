"""
Compute and cache a task's reference data (e.g. exact energies). Run once per
machine; `remote.py setup` does this on ICRN.

    uv run prepare.py --task j1j2
"""

import argparse
import importlib

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--task", default="j1j2")
args = p.parse_args()
importlib.import_module(f"tasks.{args.task}.problem").prepare()
