import unittest

import torch

from support_router.metrics import classification_metrics, expected_calibration_error


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 7.0]])
        metrics = classification_metrics(logits, torch.tensor([0, 1, 2]), num_classes=3)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_expected_calibration_error(self) -> None:
        error = expected_calibration_error(
            torch.tensor([0.8, 0.6]),
            torch.tensor([True, False]),
            num_bins=2,
        )
        self.assertAlmostEqual(error, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
