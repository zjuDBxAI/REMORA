"""REMORA method stages, shared by the public CLI and experiment runners.

The numerical and native-format helpers retain their original import paths.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in ("ours/python", "ours/test/t2i-1M/python", "ours/test/t2i-10M"):
    location = str(ROOT / folder)
    if location not in sys.path:
        sys.path.insert(0, location)
