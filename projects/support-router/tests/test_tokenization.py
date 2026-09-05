import tempfile
import unittest
from pathlib import Path

import torch

from support_router.tokenization import ScratchCollator, WordTokenizer


class WordTokenizerTests(unittest.TestCase):
    def test_fit_is_frequency_ordered_and_deterministic(self) -> None:
        tokenizer = WordTokenizer()
        tokenizer.fit(["card card cash", "Cash transfer"])

        self.assertEqual(tokenizer.vocabulary["card"], 2)
        self.assertEqual(tokenizer.vocabulary["cash"], 3)
        self.assertEqual(tokenizer.vocabulary["transfer"], 4)

    def test_unknown_tokens_and_dynamic_padding(self) -> None:
        tokenizer = WordTokenizer()
        tokenizer.fit(["known words"])
        collator = ScratchCollator(tokenizer, max_length=4)
        batch = collator(
            [
                {"text": "known mystery", "label": 2},
                {"text": "words", "label": 1},
            ]
        )

        self.assertEqual(batch["input_ids"].shape, (2, 2))
        self.assertEqual(batch["input_ids"][0, 1].item(), tokenizer.unk_id)
        self.assertTrue(torch.equal(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]])))

    def test_save_and_load_round_trip(self) -> None:
        tokenizer = WordTokenizer()
        tokenizer.fit(["cash withdrawal", "cash deposit"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = WordTokenizer.load(path)
        self.assertEqual(restored.vocabulary, tokenizer.vocabulary)


if __name__ == "__main__":
    unittest.main()
