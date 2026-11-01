"""Hash locking of scientific code/config/protocol and no-overwrite output helpers."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

FROZEN_GLOBS = ("soe/*.py", "scripts/*.py", "tests/*.py", "config.json", "PROTOCOL.md")
MANIFEST_VERSION = 1


class FreezeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frozen_files(root: Path) -> Dict[str, str]:
    root = Path(root)
    out = {}
    for pat in FROZEN_GLOBS:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def environment_versions() -> Dict[str, str]:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def write_freeze_manifest(root: Path, path: Path, approver: str, note: str = "") -> dict:
    if not approver.strip():
        raise FreezeError("an explicit approver string is required")
    files = frozen_files(root)
    for need in ("config.json", "PROTOCOL.md"):
        if need not in files:
            raise FreezeError(f"{need} missing")
    man = {
        "manifest_version": MANIFEST_VERSION,
        "approved": True,
        "approver": approver,
        "note": note,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "files": files,
        "versions": environment_versions(),
    }
    write_json_exclusive(Path(path), man)
    return man


def verify_freeze(root: Path, manifest_path: Path) -> dict:
    """Raise FreezeError unless every frozen file matches and no unlisted frozen-pattern file exists."""
    with open(manifest_path) as f:
        man = json.load(f)
    if man.get("manifest_version") != MANIFEST_VERSION or man.get("approved") is not True or not man.get("approver"):
        raise FreezeError("manifest not an approved freeze manifest")
    now = frozen_files(root)
    want = man["files"]
    missing = sorted(set(want) - set(now))
    extra = sorted(set(now) - set(want))
    changed = sorted(k for k in set(want) & set(now) if want[k] != now[k])
    if missing or extra or changed:
        raise FreezeError(f"frozen file mismatch: changed={changed} missing={missing} extra={extra}")
    ver = environment_versions()
    for k in ("python", "numpy", "torch"):
        if man["versions"].get(k) != ver[k]:
            raise FreezeError(f"version mismatch for {k}: frozen {man['versions'].get(k)} now {ver[k]}")
    return man


def create_new_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to reuse existing output directory {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json_exclusive(path: Path, obj) -> str:
    data = json.dumps(obj, indent=2, sort_keys=False, default=_default).encode()
    with open(path, "xb") as f:
        f.write(data)
    return sha256_bytes(data)


def write_bytes_exclusive(path: Path, data: bytes) -> str:
    with open(path, "xb") as f:
        f.write(data)
    return sha256_bytes(data)


def save_npz_exclusive(path: Path, **arrays) -> str:
    with open(path, "xb") as f:
        np.savez_compressed(f, **arrays)
    return sha256_file(Path(path))


def output_manifest(out_dir: Path) -> Dict[str, str]:
    out_dir = Path(out_dir)
    return {p.relative_to(out_dir).as_posix(): sha256_file(p) for p in sorted(out_dir.rglob("*")) if p.is_file()}
