#!/usr/bin/env python3
"""Public graph entry point; source/algorithm implementation stays in workflow."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'ours/workflow'))
from interface import entry

if __name__ == '__main__':
    sys.exit(entry(['build', *sys.argv[1:]]))
