import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from terraclass.cli import build_parser
from terraclass.data import DataBundle


class CliIntegrationTests(unittest.TestCase):
    def test_one_epoch_training_writes_complete_artifacts(self):
        images = torch.randn(6, 3, 64, 64)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        loader = DataLoader(TensorDataset(images, targets), batch_size=3)
        bundle = DataBundle(loader, loader, loader, ["crop", "forest", "river"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "experiment"
            args = build_parser().parse_args(
                [
                    "train",
                    "--model",
                    "cnn",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "3",
                    "--output-dir",
                    str(output_dir),
                    "--no-download",
                ]
            )
            with patch("terraclass.cli.make_dataloaders", return_value=bundle):
                args.handler(args)

            expected_files = {
                "best.pt",
                "last.pt",
                "metrics.json",
                "history.png",
                "confusion_matrix.png",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected_files)


if __name__ == "__main__":
    unittest.main()
