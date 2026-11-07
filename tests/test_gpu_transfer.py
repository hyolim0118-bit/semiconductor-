"""Transfer integrity and refusal to overwrite differing destination files."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import gpu_transfer


class TransferTests(unittest.TestCase):
    @contextmanager
    def bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = root / "artifacts"
            bundle.mkdir()
            source = bundle / "example.json"
            source.write_bytes(b'{"value": 42}\n')
            gpu_transfer.write_json(bundle / "manifest.json", {"files": [{
                "path": "artifacts/example.json", "restore_to": "outputs/example.json",
                **gpu_transfer.file_info(source)}], "external_files": []})
            with patch.object(gpu_transfer, "ROOT", root), patch.object(gpu_transfer, "BUNDLE", bundle):
                yield root, source

    def test_restore_is_idempotent(self):
        with self.bundle() as (root, source):
            gpu_transfer.restore()
            destination = root / "outputs/example.json"
            before = destination.stat().st_mtime_ns
            gpu_transfer.restore()
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(destination.stat().st_mtime_ns, before)

    def test_restore_refuses_existing_different_file(self):
        with self.bundle() as (root, source):
            destination = root / "outputs/example.json"
            destination.parent.mkdir()
            destination.write_bytes(b"previous result")
            with self.assertRaises(FileExistsError):
                gpu_transfer.restore()
            self.assertEqual(destination.read_bytes(), b"previous result")

    def test_verify_rejects_corruption(self):
        with self.bundle() as (root, source):
            source.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "mismatched"):
                gpu_transfer.verify()


if __name__ == "__main__":
    unittest.main()
