import tempfile
import unittest
from pathlib import Path

import torch

from terraclass.checkpoints import load_checkpoint, save_checkpoint
from terraclass.models import build_model


class CheckpointTests(unittest.TestCase):
    def test_round_trip(self):
        model = build_model("cnn", num_classes=2, pretrained=False)
        state = {
            "model_name": "cnn",
            "image_size": 64,
            "class_names": ["forest", "river"],
            "model_state_dict": model.state_dict(),
            "epoch": 3,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            save_checkpoint(state, path)
            restored = load_checkpoint(path, torch.device("cpu"))

        self.assertEqual(restored["model_name"], "cnn")
        self.assertEqual(restored["epoch"], 3)
        self.assertEqual(set(restored["model_state_dict"]), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
