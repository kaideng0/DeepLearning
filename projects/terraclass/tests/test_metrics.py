import unittest

import torch

from terraclass.metrics import confusion_matrix, metrics_from_confusion_matrix


class MetricsTests(unittest.TestCase):
    def test_known_predictions(self):
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        predictions = torch.tensor([0, 1, 1, 1, 2, 0])

        matrix = confusion_matrix(targets, predictions, num_classes=3)
        metrics = metrics_from_confusion_matrix(matrix)

        self.assertEqual(matrix.tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(metrics["support"], [2, 2, 2])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            confusion_matrix(torch.tensor([0]), torch.tensor([0, 1]), num_classes=2)


if __name__ == "__main__":
    unittest.main()
