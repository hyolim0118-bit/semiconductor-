"""Structural and metric checks without reading project datasets."""

import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

import torch
from torch import nn

from src.architecture_compare import DEFAULT_CONFIG, evaluate, load_config, select_models
from src.architecture_models import ResidualBlock, build_model
from src.model import MLPRegressor, count_trainable_parameters


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.config = load_config(DEFAULT_CONFIG)

    def test_selection(self):
        selected = select_models(self.config, ["A", "B", "D", "E"], full=True)
        self.assertEqual([s["id"] for s in selected], list("ABDE"))
        self.assertEqual([s["id"] for s in select_models(self.config)], list("BE"))
        self.assertEqual([s["id"] for s in select_models(self.config, full=True)], list("ABCDEF"))
        for ids in (["A", "A"], ["Z"], []):
            with self.assertRaises(ValueError):
                select_models(self.config, ids)

    def test_training_failure_keeps_successes_and_continues(self):
        from src.architecture_compare import compare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prep = root / "outputs/preprocessors"
            prep.mkdir(parents=True)
            (prep / "preprocessing_config.json").write_text(json.dumps(dict(
                target_scaling=False, feature_columns=[str(i) for i in range(226)],
                target_columns=[f"layer_{i}" for i in range(1,5)])))
            baseline = root / "outputs/training/20261107_112339_full_1f33cc10"
            baseline.mkdir(parents=True)
            (baseline / "config.json").write_text('{"source_artifacts": {}}')
            splits = root / "outputs/splits"
            splits.mkdir(parents=True)
            np.savez(str(splits / "data_split_indices.npz"), validation_indices=np.arange(81000))
            dataset = MagicMock()
            dataset.__len__.return_value = 81000
            calls = []

            def candidate(spec, config, directory, train, validation, epochs):
                calls.append(spec["id"])
                if spec["id"] == "B":
                    raise RuntimeError("deliberate test failure")
                return dict(id=spec["id"], first_epoch_seconds=1., validation_mae=2.,
                            best_selection_validation_mae=2.)

            with patch("src.architecture_compare.ROOT", root), \
                 patch("src.architecture_compare.protected_manifest", return_value={}), \
                 patch("src.architecture_compare.MappedBatches", return_value=dataset), \
                 patch("src.architecture_compare.check_model", side_effect=lambda spec, config: dict(
                     id=spec["id"], parameters=1, synthetic_step_median_seconds=1.)), \
                 patch("src.architecture_compare.train_candidate", side_effect=candidate), \
                 patch("src.architecture_compare.write_comparison"), \
                 patch("src.architecture_compare.write_report"), \
                 patch("src.architecture_analysis.analyze"), patch("builtins.print"):
                run = compare(DEFAULT_CONFIG, full=True, models=list("ABDE"))
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(calls, list("ABDE"))
            self.assertEqual([r["id"] for r in summary["results"]], list("ADE"))
            self.assertEqual([r["id"] for r in summary["failures"]], ["B"])
            self.assertIn("deliberate test failure", summary["failures"][0]["traceback"])
            self.assertTrue((run / "progress_D.json").exists())

    def test_configured_structures_and_gradients(self):
        expected = dict(A=99524, B=37572, C=280964, D=165316, E=96388, F=455940)
        for spec in self.config["models"]:
            with self.subTest(model=spec["id"]):
                model = build_model(spec)
                self.assertEqual(count_trainable_parameters(model), expected[spec["id"]])
                output = model(torch.randn(8, 226))
                self.assertEqual(tuple(output.shape), (8, 4))
                output.abs().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                model.eval()
                self.assertEqual(tuple(model(torch.randn(1, 226)).shape), (1, 4))

    def test_baseline_preserves_original_initialization_and_class(self):
        torch.manual_seed(42)
        original = MLPRegressor()
        torch.manual_seed(42)
        candidate = build_model(self.config["models"][0])
        self.assertIsInstance(candidate, MLPRegressor)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[key]))

    def test_residual_order_and_identity_gradient(self):
        block = ResidualBlock(16)
        self.assertEqual([type(module) for module in block.branch],
                         [nn.BatchNorm1d, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.Dropout])
        with torch.no_grad():
            block.branch[4].weight.zero_()
            block.branch[4].bias.zero_()
        block.eval()
        x = torch.randn(4, 16, requires_grad=True)
        output = block(x)
        self.assertTrue(torch.equal(output, x))
        output.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_batchnorm_and_dropout_modes(self):
        model = build_model(self.config["models"][4])
        x = torch.randn(32, 226)
        norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        self.assertEqual(len(norms), 3)
        model.train()
        first = model(x)
        self.assertTrue(all(m.num_batches_tracked.item() == 1 for m in norms))
        self.assertFalse(torch.equal(first, model(x)))
        model.eval()
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        self.assertTrue(torch.equal(model(x), model(x)))
        self.assertTrue(all(torch.equal(buffers[k], v) for k, v in model.named_buffers()))

    def test_metrics_with_partial_batches_and_zero_targets(self):
        class Batches:
            dataset = range(3)

            def __iter__(self):
                yield torch.zeros(2, 4), torch.tensor([[1., 2., 3., 4.]]).repeat(2, 1)
                yield torch.zeros(1, 4), torch.tensor([[1., 2., 3., 4.]])

        with tempfile.TemporaryDirectory() as temporary:
            prediction_path = Path(temporary) / "predictions.npy"
            metrics = evaluate(nn.Identity(), Batches(), prediction_path)
            predictions = np.load(str(prediction_path), allow_pickle=False)
            self.assertEqual(predictions.shape, (3, 4))
            self.assertEqual(predictions.dtype, np.float32)
            np.testing.assert_array_equal(predictions, np.zeros((3, 4)))
        self.assertEqual(metrics["validation_mae"], 2.5)
        self.assertEqual(metrics["validation_mape_percent"], 100)
        for i in range(1, 5):
            self.assertEqual(metrics["layer_{}_mae".format(i)], i)

        class ZeroBatches:
            dataset = range(1)

            def __iter__(self):
                yield torch.ones(1, 4), torch.zeros(1, 4)

        metrics = evaluate(nn.Identity(), ZeroBatches())
        self.assertEqual(metrics["mape_near_zero_counts"], [1] * 4)
        self.assertEqual(metrics["validation_mae"], 1)
        self.assertEqual(metrics["validation_mape_percent"], 1e10)


if __name__ == "__main__":
    unittest.main()

