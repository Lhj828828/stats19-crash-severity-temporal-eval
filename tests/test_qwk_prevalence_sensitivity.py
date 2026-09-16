"""Independent checks of the supplementary QWK standardization estimand."""
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from qwk_prevalence_sensitivity import standardize_confusions
from d12_bootstrap_uncertainty import metrics_from_confusions


class QWKStandardizationTests(unittest.TestCase):
    def test_matches_record_level_sklearn_with_both_marginals_updated(self):
        matrix = np.array([[17, 3, 1], [4, 9, 2], [1, 2, 3]])
        reference = np.array([0.70, 0.25, 0.05])
        truth, predicted = [], []
        for i in range(3):
            for j in range(3):
                truth.extend([i] * matrix[i, j])
                predicted.extend([j] * matrix[i, j])
        truth = np.array(truth)
        weights = reference / (matrix.sum(axis=1) / matrix.sum())
        expected = cohen_kappa_score(truth, predicted, labels=[0, 1, 2], weights="quadratic",
                                    sample_weight=weights[truth])
        adjusted = standardize_confusions(matrix, reference)
        self.assertAlmostEqual(float(metrics_from_confusions(adjusted)["qwk"]), expected, places=13)
        np.testing.assert_allclose(adjusted.sum(axis=1) / adjusted.sum(), reference)
        np.testing.assert_allclose(adjusted / adjusted.sum(axis=1)[:, None], matrix / matrix.sum(axis=1)[:, None])

    def test_no_change_when_prevalence_already_matches(self):
        matrix = np.array([[8, 2, 0], [1, 6, 3], [2, 2, 6]])
        np.testing.assert_allclose(standardize_confusions(matrix, np.ones(3) / 3), matrix)

    def test_pure_prevalence_difference_vanishes_without_changing_conditional_errors(self):
        internal = np.array([[80, 15, 5], [10, 30, 10], [1, 3, 6]])
        future = internal * np.array([1, 3, 2])[:, None]
        reference = internal.sum(axis=1) / internal.sum()
        raw = metrics_from_confusions(internal)["qwk"] - metrics_from_confusions(future)["qwk"]
        self.assertGreater(abs(float(raw)), 0.001)
        standardized = standardize_confusions(future, reference)
        self.assertAlmostEqual(float(metrics_from_confusions(internal)["qwk"]),
                               float(metrics_from_confusions(standardized)["qwk"]), places=13)
        batched = np.stack([future, future * 2])
        expected = np.stack([standardized, standardized * 2])
        np.testing.assert_allclose(standardize_confusions(batched, reference), expected)

    def test_absent_class_and_invalid_reference_are_rejected(self):
        with self.assertRaises(ValueError):
            standardize_confusions(np.diag([20, 5, 0]), np.array([0.7, 0.2, 0.1]))
        with self.assertRaises(ValueError):
            standardize_confusions(np.eye(3), np.array([0.7, 0.2, 0.2]))


if __name__ == "__main__":
    unittest.main()
