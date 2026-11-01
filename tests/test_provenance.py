import json
import tempfile
import unittest
from pathlib import Path

from tests._common import ROOT
from soe import provenance as pv


def make_tree(root: Path):
    (root / "soe").mkdir()
    (root / "tests").mkdir()
    (root / "scripts").mkdir()
    (root / "soe" / "a.py").write_text("x = 1\n")
    (root / "tests" / "t.py").write_text("y = 2\n")
    (root / "scripts" / "s.py").write_text("z = 3\n")
    (root / "config.json").write_text('{"a": 1}\n')
    (root / "PROTOCOL.md").write_text("# protocol\n")


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name) / "pkg"
        self.root.mkdir()
        make_tree(self.root)
        self.man = Path(self.td.name) / "FREEZE.json"
        pv.write_freeze_manifest(self.root, self.man, approver="unit-test")

    def tearDown(self):
        self.td.cleanup()

    def test_verify_ok(self):
        m = pv.verify_freeze(self.root, self.man)
        self.assertEqual(set(m["files"]), {"soe/a.py", "tests/t.py", "scripts/s.py", "config.json", "PROTOCOL.md"})

    def test_mutation_rejected(self):
        for rel_path in ("soe/a.py", "config.json", "PROTOCOL.md", "tests/t.py", "scripts/s.py"):
            p = self.root / rel_path
            orig = p.read_bytes()
            p.write_bytes(orig + b" ")
            with self.assertRaises(pv.FreezeError):
                pv.verify_freeze(self.root, self.man)
            p.write_bytes(orig)
            pv.verify_freeze(self.root, self.man)

    def test_extra_and_missing_rejected(self):
        extra = self.root / "soe" / "b.py"
        extra.write_text("")
        with self.assertRaises(pv.FreezeError):
            pv.verify_freeze(self.root, self.man)
        extra.unlink()
        (self.root / "tests" / "t.py").unlink()
        with self.assertRaises(pv.FreezeError):
            pv.verify_freeze(self.root, self.man)

    def test_unapproved_or_version_mismatch_rejected(self):
        m = json.loads(self.man.read_text())
        for patch in ({"approved": False}, {"approver": ""}, {"versions": {**m["versions"], "torch": "0.0"}}):
            p = Path(self.td.name) / f"m{len(str(patch))}.json"
            p.write_text(json.dumps({**m, **patch}))
            with self.assertRaises(pv.FreezeError):
                pv.verify_freeze(self.root, p)

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            pv.write_freeze_manifest(self.root, self.man, approver="again")
        with self.assertRaises(pv.FreezeError):
            pv.write_freeze_manifest(self.root, Path(self.td.name) / "x.json", approver=" ")
        d = Path(self.td.name) / "out"
        pv.create_new_dir(d)
        with self.assertRaises(FileExistsError):
            pv.create_new_dir(d)
        pv.write_json_exclusive(d / "a.json", {"a": 1})
        with self.assertRaises(FileExistsError):
            pv.write_json_exclusive(d / "a.json", {"a": 2})

    def test_runner_rejects_unfrozen_repository(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_pilot", ROOT / "scripts" / "run_pilot.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = Path(self.td.name) / "sci_out"
        # manifest of a different tree cannot validate the real repository
        with self.assertRaises(pv.FreezeError):
            mod.main(["--freeze", str(self.man), "--out", str(out)])
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
