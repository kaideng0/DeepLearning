import tempfile
import unittest
from pathlib import Path

import torch

from support_router.artifacts import load_scratch_model, save_scratch_checkpoint
from support_router.models import ScratchModelConfig, ScratchTransformerClassifier, count_parameters


def small_model() -> ScratchTransformerClassifier:
    return ScratchTransformerClassifier(
        ScratchModelConfig(
            vocab_size=20,
            num_classes=4,
            max_length=8,
            d_model=16,
            num_heads=4,
            num_layers=1,
            feedforward_size=32,
            dropout=0.0,
        )
    )


class ModelTests(unittest.TestCase):
    def test_output_shape_and_padding_mask(self) -> None:
        model = small_model().eval()
        short_ids = torch.tensor([[2, 3, 4]])
        short_mask = torch.ones_like(short_ids)
        padded_ids = torch.tensor([[2, 3, 4, 0, 0]])
        padded_mask = torch.tensor([[1, 1, 1, 0, 0]])

        with torch.inference_mode():
            short_output = model(short_ids, short_mask)
            padded_output = model(padded_ids, padded_mask)

        self.assertEqual(short_output.shape, (1, 4))
        torch.testing.assert_close(short_output, padded_output)

    def test_checkpoint_round_trip(self) -> None:
        model = small_model().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_scratch_checkpoint(model, path, epoch=2, best_validation_f1=0.5)
            restored = load_scratch_model(path, torch.device("cpu"))
        self.assertEqual(count_parameters(restored), count_parameters(model))
        for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
