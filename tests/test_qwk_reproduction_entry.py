"""Guard the additive QWK entry against overwrites and unsafe directory choices."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce_qwk_sensitivity as entry


class QWKReproductionEntryTests(unittest.TestCase):
    def test_input_inventory_is_complete_and_excludes_models(self):
        names = entry.input_names()
        self.assertEqual(len(names), 28)
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(sum(n.endswith(".npz") for n in names), 20)
        self.assertFalse(any(n.startswith(("models/", "data/raw/")) for n in names))

    def test_workspace_must_be_separate_from_parent_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parent, source = base / "parent", base / "source"
            parent.mkdir()
            source.mkdir()
            for bad in (base, parent, parent / "nested", source, source / "nested"):
                with self.assertRaises(ValueError):
                    entry.check_paths(parent, bad, source)
            self.assertEqual(entry.check_paths(parent, base / "new", source)[1], (base / "new").resolve())

    def test_existing_workspace_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with self.assertRaises(FileExistsError):
                entry.prepare(p, p)


if __name__ == "__main__":
    unittest.main()
