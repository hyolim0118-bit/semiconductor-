"""STEP 8 forward-only checks; existing artifacts are opened read-only."""

import json
from pathlib import Path
import platform
import unittest

import numpy as np
import torch
from torch import nn

from src.config import HIDDEN_DIMS, INPUT_DIM, OUTPUT_DIM
from src.model import MLPRegressor, count_trainable_parameters


ROOT = Path(__file__).resolve().parents[1]


def artifact_snapshot():
    paths = list(ROOT.glob("*.csv"))
    paths.extend(path for path in (ROOT / "outputs").rglob("*") if path.is_file())
    return {str(path.relative_to(ROOT)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in paths}


class ModelForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = artifact_snapshot()
        torch.manual_seed(42)
        cls.model = MLPRegressor().eval()
        print("Python:", platform.python_version())
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA build:", torch.version.cuda)
        print("Test device: cpu")
        for index in range(torch.cuda.device_count()):
            print("CUDA device {}: {}".format(index, torch.cuda.get_device_name(index)))
        print(cls.model)
        print("Trainable parameters:", count_trainable_parameters(cls.model))

    @classmethod
    def tearDownClass(cls):
        if artifact_snapshot() != cls.before:
            raise AssertionError("Existing CSV/output file sizes or mtimes changed")
        print("Existing CSV/output file sizes and mtimes: unchanged")

    def test_architecture(self):
        self.assertEqual((INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM),
                         (226, (256, 128, 64), 4))
        self.assertEqual([type(layer) for layer in self.model.network],
                         [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU,
                          nn.Linear, nn.ReLU, nn.Linear])
        self.assertEqual(count_trainable_parameters(self.model), 99524)

    def test_random_batches(self):
        with torch.no_grad():
            for batch_size in (32, 1, 1024):
                with self.subTest(batch_size=batch_size):
                    output = self.model(torch.randn(batch_size, INPUT_DIM,
                                                    dtype=torch.float32))
                    self.assertEqual(tuple(output.shape), (batch_size, OUTPUT_DIM))
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(output).all().item())
                    print("Random batch {} -> {}: finite".format(
                        batch_size, tuple(output.shape)))

    def test_invalid_shapes(self):
        for shape in ((32, 225), (32, 227), (226,), (2, 3, 226), ()):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "Expected input shape") as caught:
                    self.model(torch.zeros(shape))
                print("Rejected:", caught.exception)
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            self.model(np.zeros((1, INPUT_DIM), dtype=np.float32))

    def test_preprocessed_sample(self):
        path = ROOT / "outputs" / "preprocessed" / "X_train_scaled.npy"
        if not path.is_file():
            self.skipTest("X_train_scaled.npy is missing")
        with (ROOT / "outputs" / "preprocessors" /
              "preprocessing_config.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        self.assertEqual(metadata["feature_columns"], [str(i) for i in range(INPUT_DIM)])
        self.assertEqual(metadata["target_columns"],
                         ["layer_{}".format(i) for i in range(1, OUTPUT_DIM + 1)])
        values = np.load(str(path), mmap_mode="r", allow_pickle=False)
        self.assertEqual(values.ndim, 2)
        self.assertEqual(values.shape[1], INPUT_DIM)
        self.assertEqual(values.dtype, np.float32)
        self.assertGreater(len(values), 0)
        # Copy only 32 rows: the read-only mapping is never exposed for writing.
        sample = torch.from_numpy(np.array(values[:32], copy=True))
        self.assertTrue(torch.isfinite(sample).all().item())
        with torch.no_grad():
            output = self.model(sample)
        self.assertEqual(tuple(output.shape), (len(sample), OUTPUT_DIM))
        self.assertTrue(torch.isfinite(output).all().item())
        print("Saved X: shape={}, dtype={}, mmap_mode=r".format(values.shape, values.dtype))
        print("Real sample {} -> {}: finite".format(tuple(sample.shape), tuple(output.shape)))
        print("Target scaling:", metadata["target_scaling"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
