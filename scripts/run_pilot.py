"""SCIENTIFIC runner. Refuses to run without a matching approved freeze manifest.

    python scripts/run_pilot.py --freeze FREEZE_MANIFEST.json --out runs/scientific_<label>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _path import ROOT  # noqa: E402  (pins BLAS threads before numpy loads)
import argparse  # noqa: E402
import json  # noqa: E402
from soe.provenance import sha256_file, verify_freeze  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    man = verify_freeze(ROOT, Path(a.freeze))  # raises FreezeError before anything is created
    man = {**man, "manifest_path": str(Path(a.freeze).resolve()), "manifest_sha256": sha256_file(Path(a.freeze))}
    from soe import pipeline
    with open(ROOT / "config.json") as f:
        cfg = json.load(f)
    n_env = len(cfg["teacher"]["cells_order"]) * cfg["teacher"]["replicate_blocks_per_cell"] * len(cfg["covariance"]["kinds_order"])
    summ = pipeline.run(cfg, root=cfg["seeds"]["scientific_root"], env_ids=list(range(n_env)), out_dir=Path(a.out),
                        mode="scientific", run_tests=True, freeze_manifest=man, analyze=True)
    print(json.dumps({k: summ.get(k) for k in ("complete", "valid", "total_runtime_s")}, indent=2))
    print("VERDICT:", summ.get("verdict", {}).get("verdict"))


if __name__ == "__main__":
    main()
