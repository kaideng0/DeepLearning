"""From-scratch and pretrained transformer intent classifiers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

MODEL_TYPES = ("scratch", "distilbert")
DEFAULT_BASE_MODEL = "distilbert/distilbert-base-uncased"


@dataclass(frozen=True)
class ScratchModelConfig:
    vocab_size: int
    num_classes: int
    pad_id: int = 0
    max_length: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feedforward_size: int = 256
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScratchTransformerClassifier(nn.Module):
    """A compact encoder-only transformer with masked mean pooling."""

    def __init__(self, config: ScratchModelConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("sequence exceeds configured max_length")

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        embeddings = embeddings + self.position_embedding(positions)
        padding_mask = ~attention_mask.bool()
        encoded = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        encoded = self.final_norm(encoded)

        valid_tokens = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


def build_pretrained_classifier(
    base_model: str,
    *,
    num_classes: int,
    class_names: list[str],
) -> nn.Module:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face Transformers is required for DistilBERT experiments."
        ) from error
    id_to_label = {index: label for index, label in enumerate(class_names)}
    label_to_id = {label: index for index, label in id_to_label.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_classes,
        id2label=id_to_label,
        label2id=label_to_id,
    )


def extract_logits(model_output: Any) -> Tensor:
    return model_output.logits if hasattr(model_output, "logits") else model_output


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )
