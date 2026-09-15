"""Relocation must preserve mathematical source identity, not hide source edits."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


class ModelFactoryRelocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.factory = self.root / "factory"
        self.folder = self.factory / "TabUF"
        (self.folder / "figures").mkdir(parents=True)
        (self.folder / "main.tex").write_text(r"\input{figures/operator}")
        (self.folder / "figures/operator.tex").write_text("original math")
        script = Path(__file__).resolve().parents[2] / "scripts/build_model_source_manifest.py"
        spec = importlib.util.spec_from_file_location("source_manifest_relocation", script)
        assert spec is not None and spec.loader is not None
        self.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.builder)
        self.builder.FACTORY = self.factory
        key = "CONTRACTS" if hasattr(self.builder, "CONTRACTS") else "ENTRYPOINTS"
        setattr(self.builder, key, {"sample": "TabUF/main.tex"})

    def relocate(self) -> Path:
        target = self.factory / "first-generation-models/TabUF"
        target.parent.mkdir()
        self.folder.rename(target)
        self.folder.symlink_to("first-generation-models/TabUF", target_is_directory=True)
        return target

    def test_relocation_preserves_closure_but_math_edit_changes_hash(self) -> None:
        before = self.builder.source_closure(self.folder / "main.tex")
        before_hash = self.builder.content_hash(before)
        target = self.relocate()
        after = self.builder.source_closure(target / "main.tex")
        self.assertEqual(before, after)
        self.assertEqual(before_hash, self.builder.content_hash(after))
        self.assertEqual(self.builder.logical(target / "main.tex"), "TabUF/main.tex")
        (target / "figures/operator.tex").write_text("changed math")
        changed = self.builder.source_closure(self.folder / "main.tex")
        self.assertNotEqual(before_hash, self.builder.content_hash(changed))
        self.assertEqual(set(before), set(changed))

    def test_retired_alias_preserves_historical_closure(self) -> None:
        key = "CONTRACTS" if hasattr(self.builder, "CONTRACTS") else "ENTRYPOINTS"
        setattr(self.builder, key, {"graph": "TabU4Graph/main.tex"})
        old = self.factory / "TabU4Graph"
        self.folder.rename(old)
        before = self.builder.source_closure(old / "main.tex")
        target = self.factory / "first-generation-models/TabU4Graph"
        target.parent.mkdir()
        old.rename(target)
        self.assertFalse(old.exists())
        self.assertEqual(self.builder.physical(old / "main.tex"), (target / "main.tex").resolve())
        self.assertEqual(before, self.builder.source_closure(old / "main.tex"))
        # A wrong old file must remain visible to hash verification, not be hidden.
        old.mkdir()
        (old / "main.tex").write_text("conflicting old source")
        self.assertEqual(self.builder.physical(old / "main.tex"), (old / "main.tex").resolve())

    def test_v2_without_alias_preserves_closure(self) -> None:
        key = "CONTRACTS" if hasattr(self.builder, "CONTRACTS") else "ENTRYPOINTS"
        setattr(self.builder, key, {"v2": "TabU-v2/main.tex"})
        old = self.factory / "TabU-v2"
        self.folder.rename(old)
        before = self.builder.source_closure(old / "main.tex")
        target = self.factory / "table-cell-as-query-models/TabU-v2"
        target.parent.mkdir()
        old.rename(target)
        self.assertFalse(old.exists())
        self.assertEqual(before, self.builder.source_closure(old / "main.tex"))

    def test_unregistered_source_uses_its_own_path(self) -> None:
        self.relocate()
        new = self.factory / "TabU-TAR/new.tex"
        new.parent.mkdir()
        new.write_text("new contract")
        self.assertEqual(self.builder.logical(new), "TabU-TAR/new.tex")

    def test_symlink_include_cannot_escape_factory(self) -> None:
        outside = self.root / "outside.tex"
        outside.write_text("external source")
        (self.folder / "outside.tex").symlink_to(outside)
        (self.folder / "main.tex").write_text(r"\input{outside}")
        with self.assertRaisesRegex(ValueError, "escapes model-factory"):
            self.builder.source_closure(self.folder / "main.tex")


if __name__ == "__main__":
    unittest.main()
