"""Verify independent restoration and LR override after loading Adam."""

from pathlib import Path
import tempfile
import unittest

import torch

from src.lr_compare import restore
from src.model import MLPRegressor
from src.train import SETTINGS


class ResumeTests(unittest.TestCase):
    def test_independent_model_and_adam_restoration(self):
        model = MLPRegressor()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model(torch.zeros(2, 226)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save(dict(epoch=200, config=SETTINGS, validation_mae=1.0,
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict()), str(path))
            first, first_opt, _ = restore(path, 0.0003)
            with torch.no_grad():
                next(first.parameters()).add_(100)
            next(iter(first_opt.state.values()))["exp_avg"].add_(100)
            second, second_opt, _ = restore(path, 0.0001)
            self.assertEqual(second_opt.param_groups[0]["lr"], 0.0001)
            self.assertEqual(first_opt.param_groups[0]["lr"], 0.0003)
            for original, restored in zip(model.parameters(), second.parameters()):
                self.assertTrue(torch.equal(original, restored))
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(optimizer.state[original][key], second_opt.state[restored][key]))


if __name__ == "__main__":
    unittest.main()
