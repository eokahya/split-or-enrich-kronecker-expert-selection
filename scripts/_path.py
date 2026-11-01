"""Imported FIRST by every script: pins BLAS/OpenMP threads (before numpy loads) and the import path."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.json") as _f:
    for _k, _v in json.load(_f)["runtime"]["blas_threads_env"].items():
        os.environ.setdefault(_k, _v)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
