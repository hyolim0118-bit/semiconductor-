"""Pure synthetic checks; no project datasets or checkpoints are opened."""

import unittest

import numpy as np

from src.final_candidates import FrozenCandidates, comparison_rows, metrics


class FinalCandidateTests(unittest.TestCase):
    def test_standard_mape_and_zero_rejection(self):
        result = metrics(np.array([-10., 20.]), np.array([0., 10.]))
        self.assertEqual(result["mae"], 10.)
        self.assertEqual(result["mape_percent"], 75.)
        with self.assertRaises(ValueError):
            metrics(np.array([0.]), np.array([1.]))

    def test_ranges_counts_and_improvements(self):
        target = np.array([[10., 100., 110., 200.], [210., 300., 10., 300.]])
        predictions = dict(single_mlp=target+1, lr_mean_ensemble=target+.5)
        overall, ranges = comparison_rows(target, predictions)
        self.assertEqual(overall[1]["mae_improvement_percent"], 50.)
        self.assertAlmostEqual(overall[1]["mape_percent_improvement_percent"], 50.)
        pooled = [r for r in ranges if r["scope"] == "overall" and r["candidate"] == "single_mlp"]
        self.assertEqual([r["target_count"] for r in pooled], [3, 2, 3])
        target[0, 0] = 105
        with self.assertRaises(ValueError):
            comparison_rows(target, predictions)

    def test_predict_fixed_mean_partial_batch_and_order(self):
        predictor = FrozenCandidates.__new__(FrozenCandidates)
        predictor.manifest = dict(candidates=dict(
            single_mlp=dict(members=["b"]), lr_mean_ensemble=dict(members=["a", "b", "c"])))
        predictor.models = {"a": lambda x: x[:, :4]+1,
                            "b": lambda x: x[:, :4]+2,
                            "c": lambda x: x[:, :4]+6}
        x = np.arange(3*226, dtype=np.float32).reshape(3, 226)
        np.testing.assert_array_equal(predictor.predict("single_mlp", x, 2), x[:, :4]+2)
        np.testing.assert_array_equal(predictor.predict("lr_mean_ensemble", x, 2), x[:, :4].astype(np.float64)+3)


if __name__ == "__main__":
    unittest.main()
