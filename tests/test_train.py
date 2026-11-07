"""Small synthetic checks for batching and MAE aggregation, without real training."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from src.train import MappedBatches, make_loader, run_epoch


class TrainingTests(unittest.TestCase):
    def test_mapping_and_shuffled_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            x = np.zeros((1025, 226), dtype=np.float32)
            x[:, 0] = np.arange(1025)
            y = np.zeros((1025, 4), dtype=np.float32)
            np.save(str(path / "X_train_scaled.npy"), x)
            np.save(str(path / "y_train.npy"), y)
            dataset = MappedBatches(path, "train", 1025)
            self.assertFalse(dataset.x.flags.writeable)
            batches = list(make_loader(dataset, True, torch.Generator().manual_seed(42)))
            self.assertEqual([len(batch[0]) for batch in batches], [1024, 1])
            seen = torch.cat([batch[0][:, 0] for batch in batches]).numpy()
            np.testing.assert_array_equal(np.sort(seen), np.arange(1025))
            batches[0][0].fill_(-1)
            np.testing.assert_array_equal(dataset.x, x)
            dataset.x._mmap.close()
            dataset.y._mmap.close()

    def test_mae_weights_partial_batch_by_element_count(self):
        class Batches:
            dataset = range(3)

            def __iter__(self):
                yield torch.zeros(2, 4), torch.ones(2, 4)
                yield torch.zeros(1, 4), torch.full((1, 4), 4.0)

        model = nn.Identity()
        mae, seconds = run_epoch(model, Batches(), nn.L1Loss())
        self.assertEqual(mae, 2.0)
        self.assertGreaterEqual(seconds, 0)
        self.assertFalse(model.training)

    def test_nonfinite_batch_is_rejected(self):
        class Batches:
            dataset = range(1)

            def __iter__(self):
                yield torch.full((1, 4), float("nan")), torch.zeros(1, 4)

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            run_epoch(nn.Identity(), Batches(), nn.L1Loss())


if __name__ == "__main__":
    unittest.main()
