"""Write the freeze manifest (coordinator only, after final protocol approval).

    python scripts/freeze.py --approver "<name / message id>" --out FREEZE_MANIFEST.json
"""
import argparse

from _path import ROOT
from soe.provenance import write_freeze_manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--approver", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    man = write_freeze_manifest(ROOT, a.out, a.approver, a.note)
    print(f"froze {len(man['files'])} files -> {a.out}")


if __name__ == "__main__":
    main()
