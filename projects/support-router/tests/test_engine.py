import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from support_router.data import TextIntentDataset
from support_router.engine import run_epoch
from support_router.models import ScratchModelConfig, ScratchTransformerClassifier
from support_router.tokenization import ScratchCollator, WordTokenizer


class EngineTests(unittest.TestCase):
    def test_training_epoch_updates_parameters(self) -> None:
        dataset = TextIntentDataset(
            ["cash withdrawal", "cash deposit", "lost card", "new card"],
            [0, 0, 1, 1],
        )
        tokenizer = WordTokenizer()
        tokenizer.fit(dataset.texts)
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=ScratchCollator(tokenizer, max_length=4),
        )
        model = ScratchTransformerClassifier(
            ScratchModelConfig(
                vocab_size=len(tokenizer),
                num_classes=2,
                max_length=4,
                d_model=8,
                num_heads=2,
                num_layers=1,
                feedforward_size=16,
                dropout=0.0,
            )
        )
        before = model.classifier.weight.detach().clone()
        metrics = run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            num_classes=2,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
        )

        self.assertFalse(torch.equal(before, model.classifier.weight))
        self.assertIn("macro_f1", metrics)


if __name__ == "__main__":
    unittest.main()
