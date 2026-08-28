import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from terraclass.engine import run_epoch


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.fc(inputs)


class EngineTests(unittest.TestCase):
    def setUp(self):
        inputs = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.8],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        targets = torch.tensor([0, 0, 1, 1])
        self.loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    def test_evaluation_returns_expected_fields(self):
        metrics = run_epoch(
            TinyClassifier(),
            self.loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )

        self.assertIn("loss", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)

    def test_training_updates_parameters(self):
        model = TinyClassifier()
        before = model.fc.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        run_epoch(
            model,
            self.loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            optimizer=optimizer,
        )

        self.assertFalse(torch.equal(before, model.fc.weight))


if __name__ == "__main__":
    unittest.main()
