"""Measure one child (including its children) without a GNU time dependency."""
import os
from pathlib import Path
import subprocess
import sys
import time

if __name__ == "__main__":
    started=time.monotonic()
    child=subprocess.Popen(sys.argv[2:])
    _,status,usage=os.wait4(child.pid,0)
    child.returncode=os.waitstatus_to_exitcode(status)
    code=child.returncode if child.returncode>=0 else 128-child.returncode
    with Path(sys.argv[1]).open("w") as f:
        f.write(f"{time.monotonic()-started},{usage.ru_maxrss},{code}\n"); f.flush(); os.fsync(f.fileno())
    raise SystemExit(code)
