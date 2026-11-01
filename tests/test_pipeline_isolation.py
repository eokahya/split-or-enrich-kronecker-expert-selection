import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._common import load_cfg
from soe import pipeline


class TestPipelineIsolation(unittest.TestCase):
    def test_selections_locked_before_any_nonpilot_split(self):
        cfg = load_cfg()
        cfg = copy.deepcopy(cfg)
        cfg["training"]["steps"] = 60
        cfg["early_selector"]["primary_common_ceiling_s"] = 0.05
        events = []
        real = pipeline.sample_split
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run"

            def spy(c, env, split, n=None):
                events.append((split, (out / "selections.sha256").exists()))
                return real(c, env, split, n)

            with mock.patch.object(pipeline, "sample_split", spy):
                summ = pipeline.run(cfg, root=cfg["seeds"]["unit_test_root"], env_ids=[0, 1], out_dir=out,
                                    mode="technical", run_tests=False, analyze=True, log=lambda s: None)
            pilot_calls = [e for e in events if e[0] == "pilot"]
            other = [e for e in events if e[0] != "pilot"]
            self.assertEqual(len(other), 6)
            self.assertTrue(all(locked for _, locked in other))
            # pilot calls for the two environments happen before the lock (warmup pilot excluded: it is first)
            self.assertTrue(all(not locked for _, locked in pilot_calls))
            sel = json.loads((out / "selections.json").read_text())
            self.assertEqual(set(sel), {"0", "1"})
            self.assertTrue(summ["numerical_gates"]["moment_vs_sample_ok"])
            self.assertTrue(summ["numerical_gates"]["two_impl_ok"])
            self.assertTrue(summ["numerical_gates"]["finite"])
            with self.assertRaises(FileExistsError):
                pipeline.run(cfg, root=cfg["seeds"]["unit_test_root"], env_ids=[0], out_dir=out, mode="technical",
                             run_tests=False, log=lambda s: None)

    def test_scientific_mode_guards(self):
        cfg = load_cfg()
        fake_manifest = {"files": {}, "manifest_path": "/nonexistent"}
        with tempfile.TemporaryDirectory() as td:
            sci = cfg["seeds"]["scientific_root"]
            # subset of environments
            with self.assertRaises(RuntimeError):
                pipeline.run(cfg, root=sci, env_ids=[0, 1], out_dir=Path(td) / "a", mode="scientific",
                             run_tests=True, freeze_manifest=fake_manifest)
            # programmatically modified config
            c2 = copy.deepcopy(cfg)
            c2["training"]["steps"] = 10
            with self.assertRaises(RuntimeError):
                pipeline.run(c2, root=sci, env_ids=list(range(24)), out_dir=Path(td) / "b", mode="scientific",
                             run_tests=True, freeze_manifest=fake_manifest)
            # config hash not in manifest
            with self.assertRaises(RuntimeError):
                pipeline.run(cfg, root=sci, env_ids=list(range(24)), out_dir=Path(td) / "c", mode="scientific",
                             run_tests=True, freeze_manifest=fake_manifest)
            self.assertFalse(any((Path(td) / n).exists() for n in "abc"))

    def test_failing_tests_abort_before_any_data(self):
        cfg = load_cfg()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run"
            with mock.patch.object(pipeline, "run_unit_tests", lambda: {"ok": False, "ran": 1}), \
                    mock.patch.object(pipeline, "sample_split", side_effect=AssertionError("data generated")):
                summ = pipeline.run(cfg, root=cfg["seeds"]["unit_test_root"], env_ids=[0], out_dir=out,
                                    mode="technical", run_tests=True, log=lambda s: None)
            self.assertEqual(summ["verdict"]["verdict"], "INVALID")
            self.assertEqual(list((out / "pilot").iterdir()), [])

    def test_scientific_root_guard(self):
        cfg = load_cfg()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                pipeline.run(cfg, root=cfg["seeds"]["scientific_root"], env_ids=[0], out_dir=Path(td) / "x",
                             mode="technical", run_tests=False)
            with self.assertRaises(RuntimeError):
                pipeline.run(cfg, root=cfg["seeds"]["scientific_root"], env_ids=[0], out_dir=Path(td) / "y",
                             mode="scientific", run_tests=False, freeze_manifest=None)
            self.assertFalse((Path(td) / "x").exists())


if __name__ == "__main__":
    unittest.main()
