"""Shared test helpers and DECLARED tolerances (fixed before running tests; never loosened after a failure)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Declared tolerances
TOL_ALG = 1e-10        # float64 algebraic identities (relative, scale >= 1)
TOL_FD = 1e-6          # central finite differences vs autograd (relative), step h = 1e-6
FD_H = 1e-6
MC_SE = 5.0            # Monte Carlo agreement in standard errors
TOL_MPS32 = 1e-4       # float32 (MPS or CPU) vs CPU float64 relative agreement on losses

UNIT_ROOT = 7700123


def load_cfg():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def rel(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))
