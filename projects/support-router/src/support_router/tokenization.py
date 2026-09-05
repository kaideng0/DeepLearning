"""A small word tokenizer used by the from-scratch transformer."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", re.IGNORECASE)


class WordTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, vocabulary: dict[str, int] | None = None) -> None:
        self.vocabulary = vocabulary or {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self._validate_vocabulary()

    @property
    def pad_id(self) -> int:
        return self.vocabulary[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocabulary[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.vocabulary)

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def fit(
        self,
        texts: Iterable[str],
        *,
        min_frequency: int = 1,
        max_vocabulary_size: int = 20_000,
    ) -> None:
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if max_vocabulary_size < 2:
            raise ValueError("max_vocabulary_size must leave room for special tokens")
        counts = Counter(token for text in texts for token in self.tokenize(text))
        candidates = [
            token
            for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_frequency
        ]
        candidates = candidates[: max_vocabulary_size - 2]
        self.vocabulary = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.vocabulary.update({token: index + 2 for index, token in enumerate(candidates)})

    def encode(self, text: str, *, max_length: int) -> list[int]:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        tokens = self.tokenize(text)[:max_length]
        if not tokens:
            return [self.unk_id]
        return [self.vocabulary.get(token, self.unk_id) for token in tokens]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"vocabulary": self.vocabulary}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        vocabulary = {
            str(token): int(index)
            for token, index in value["vocabulary"].items()
        }
        return cls(vocabulary=vocabulary)

    def _validate_vocabulary(self) -> None:
        pad_id = self.vocabulary.get(self.PAD_TOKEN)
        unk_id = self.vocabulary.get(self.UNK_TOKEN)
        if pad_id != 0 or unk_id != 1:
            raise ValueError("Vocabulary must assign <pad>=0 and <unk>=1")


class ScratchCollator:
    """Tokenize and dynamically pad a batch of text examples."""

    def __init__(self, tokenizer: WordTokenizer, *, max_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
        encoded = [
            self.tokenizer.encode(str(example["text"]), max_length=self.max_length)
            for example in examples
        ]
        batch_length = max(len(tokens) for tokens in encoded)
        input_ids = torch.full(
            (len(examples), batch_length),
            self.tokenizer.pad_id,
            dtype=torch.int64,
        )
        attention_mask = torch.zeros((len(examples), batch_length), dtype=torch.int64)
        for row, tokens in enumerate(encoded):
            length = len(tokens)
            input_ids[row, :length] = torch.tensor(tokens, dtype=torch.int64)
            attention_mask[row, :length] = 1
        labels = torch.tensor([int(example["label"]) for example in examples], dtype=torch.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class PretrainedCollator:
    """Batch raw text with a Hugging Face tokenizer at runtime."""

    def __init__(self, tokenizer: object, *, max_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(  # type: ignore[operator]
            [str(example["text"]) for example in examples],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [int(example["label"]) for example in examples],
            dtype=torch.int64,
        )
        return dict(encoded)
