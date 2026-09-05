"""Command-line training, evaluation, and inference for SupportRouter."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from support_router.artifacts import (
    load_metadata,
    load_scratch_model,
    save_metadata,
    save_scratch_checkpoint,
)
from support_router.data import IntentSplits, TextIntentDataset, load_banking77
from support_router.engine import run_epoch
from support_router.models import (
    DEFAULT_BASE_MODEL,
    MODEL_TYPES,
    ScratchModelConfig,
    ScratchTransformerClassifier,
    build_pretrained_classifier,
    count_parameters,
)
from support_router.plots import plot_confusion_matrix, plot_history
from support_router.tokenization import PretrainedCollator, ScratchCollator, WordTokenizer
from support_router.utils import describe_device, resolve_device, save_json, seed_everything


def _load_auto_tokenizer(source: str | Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face Transformers is required for DistilBERT experiments."
        ) from error
    return AutoTokenizer.from_pretrained(str(source))


def _load_pretrained_model(source: str | Path, device: torch.device) -> nn.Module:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face Transformers is required for DistilBERT experiments."
        ) from error
    return AutoModelForSequenceClassification.from_pretrained(str(source)).to(device).eval()


def _data_loader(
    dataset: TextIntentDataset,
    collator: object,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _metric_summary(name: str, metrics: dict[str, object]) -> str:
    return (
        f"{name}: loss={float(metrics['loss']):.4f}, "
        f"accuracy={float(metrics['accuracy']):.4f}, "
        f"macro_f1={float(metrics['macro_f1']):.4f}"
    )


def _save_classification_report(
    metrics: dict[str, object],
    class_names: Sequence[str],
    destination: Path,
) -> None:
    precision = metrics["per_class_precision"]
    recall = metrics["per_class_recall"]
    f1 = metrics["per_class_f1"]
    support = metrics["support"]
    lines = [
        "# Test classification report",
        "",
        "| Intent | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for index, class_name in enumerate(class_names):
        lines.append(
            f"| `{class_name}` | {precision[index]:.3f} | {recall[index]:.3f} | "
            f"{f1[index]:.3f} | {support[index]} |"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_training_components(
    args: argparse.Namespace,
    splits: IntentSplits,
    output_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, object, dict[str, Any]]:
    if args.model == "scratch":
        tokenizer = WordTokenizer()
        tokenizer.fit(
            splits.train.texts,
            min_frequency=args.min_frequency,
            max_vocabulary_size=args.max_vocabulary_size,
        )
        tokenizer.save(output_dir / "tokenizer.json")
        config = ScratchModelConfig(
            vocab_size=len(tokenizer),
            num_classes=len(splits.class_names),
            pad_id=tokenizer.pad_id,
            max_length=args.max_length,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
        )
        model = ScratchTransformerClassifier(config).to(device)
        collator = ScratchCollator(tokenizer, max_length=args.max_length)
        model_details: dict[str, Any] = {"scratch_config": config.to_dict()}
    else:
        tokenizer = _load_auto_tokenizer(args.base_model)
        model = build_pretrained_classifier(
            args.base_model,
            num_classes=len(splits.class_names),
            class_names=splits.class_names,
        ).to(device)
        collator = PretrainedCollator(tokenizer, max_length=args.max_length)
        model_details = {"base_model": args.base_model}
    return model, collator, model_details


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"CUDA available: {'yes' if torch.cuda.is_available() else 'no'}")
    print(f"Training device: {describe_device(device)}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_banking77(
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        dataset_name=args.dataset,
    )
    print(
        f"Dataset: {len(splits.train):,} train / {len(splits.validation):,} validation / "
        f"{len(splits.test):,} test; {len(splits.class_names)} intents"
    )

    epochs = args.epochs or (20 if args.model == "scratch" else 4)
    batch_size = args.batch_size or (64 if args.model == "scratch" else 16)
    learning_rate = args.learning_rate or (1e-3 if args.model == "scratch" else 2e-5)
    model, collator, model_details = _build_training_components(
        args, splits, output_dir, device
    )
    print(
        f"Parameters: {count_parameters(model):,} total / "
        f"{count_parameters(model, trainable_only=True):,} trainable"
    )

    metadata: dict[str, Any] = {
        "project": "SupportRouter",
        "model_type": args.model,
        "class_names": splits.class_names,
        "dataset": args.dataset,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "max_length": args.max_length,
        "confidence_threshold": args.confidence_threshold,
        **model_details,
    }
    save_metadata(metadata, output_dir)

    train_loader = _data_loader(
        splits.train,
        collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    validation_loader = _data_loader(
        splits.validation,
        collator,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    history: list[dict[str, object]] = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    best_model_dir = output_dir / "best_model"
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes=len(splits.class_names),
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            num_classes=len(splits.class_names),
        )
        validation_f1 = float(validation_metrics["macro_f1"])
        scheduler.step(validation_f1)
        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        save_json(history, output_dir / "history.json")
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | {_metric_summary('train', train_metrics)} | "
            f"{_metric_summary('validation', validation_metrics)}"
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            epochs_without_improvement = 0
            if args.model == "scratch":
                save_scratch_checkpoint(
                    model,
                    output_dir / "best.pt",
                    epoch=epoch,
                    best_validation_f1=best_f1,
                )
            else:
                model.save_pretrained(best_model_dir)  # type: ignore[attr-defined]
                collator.tokenizer.save_pretrained(best_model_dir)  # type: ignore[attr-defined]
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    plot_history(history, output_dir / "learning_curves.png")
    if args.model == "scratch":
        best_model = load_scratch_model(output_dir / "best.pt", device)
    else:
        best_model = _load_pretrained_model(best_model_dir, device)
    test_loader = _data_loader(
        splits.test,
        collator,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    test_metrics = run_epoch(
        best_model,
        test_loader,
        criterion,
        device,
        num_classes=len(splits.class_names),
    )
    save_json(test_metrics, output_dir / "test_metrics.json")
    _save_classification_report(
        test_metrics, splits.class_names, output_dir / "test_report.md"
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        splits.class_names,
        output_dir / "confusion_matrix.png",
    )
    print(_metric_summary("test", test_metrics))
    print(f"Artifacts saved to {output_dir}")


def _load_experiment(
    experiment_dir: Path, device: torch.device
) -> tuple[nn.Module, object, dict[str, Any]]:
    metadata = load_metadata(experiment_dir)
    max_length = int(metadata["max_length"])
    if metadata["model_type"] == "scratch":
        tokenizer = WordTokenizer.load(experiment_dir / "tokenizer.json")
        model = load_scratch_model(experiment_dir / "best.pt", device)
        collator = ScratchCollator(tokenizer, max_length=max_length)
    elif metadata["model_type"] == "distilbert":
        model_dir = experiment_dir / "best_model"
        tokenizer = _load_auto_tokenizer(model_dir)
        model = _load_pretrained_model(model_dir, device)
        collator = PretrainedCollator(tokenizer, max_length=max_length)
    else:
        raise ValueError(f"Unsupported model type: {metadata['model_type']}")
    return model, collator, metadata


def evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    experiment_dir = Path(args.experiment_dir).resolve()
    model, collator, metadata = _load_experiment(experiment_dir, device)
    splits = load_banking77(
        validation_ratio=float(metadata["validation_ratio"]),
        seed=int(metadata["seed"]),
        dataset_name=str(metadata["dataset"]),
    )
    loader = _data_loader(
        splits.test,
        collator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=int(metadata["seed"]),
    )
    metrics = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        num_classes=len(metadata["class_names"]),
    )
    print(_metric_summary("test", metrics))
    print(
        f"top_3_accuracy={float(metrics['top_3_accuracy']):.4f}, "
        f"calibration_error={float(metrics['expected_calibration_error']):.4f}"
    )


def predict(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    experiment_dir = Path(args.experiment_dir).resolve()
    model, collator, metadata = _load_experiment(experiment_dir, device)
    examples = [{"text": text, "label": 0} for text in args.text]
    batch = collator(examples)  # type: ignore[operator]
    with torch.inference_mode():
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        logits = output.logits if hasattr(output, "logits") else output
        probabilities = logits.softmax(dim=1).cpu()

    class_names = metadata["class_names"]
    threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else float(metadata.get("confidence_threshold", 0.65))
    )
    results = []
    for text, distribution in zip(args.text, probabilities, strict=True):
        values, indices = distribution.topk(min(args.top_k, len(class_names)))
        confidence = float(values[0])
        results.append(
            {
                "text": text,
                "predicted_intent": class_names[int(indices[0])],
                "confidence": round(confidence, 6),
                "route_to_human": confidence < threshold,
                "top_predictions": [
                    {"intent": class_names[int(index)], "probability": round(float(value), 6)}
                    for value, index in zip(values, indices, strict=True)
                ],
            }
        )
    print(json.dumps(results, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-router",
        description="Train and use Transformer-based Banking77 intent classifiers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and test a model")
    train_parser.add_argument("--model", choices=MODEL_TYPES, default="scratch")
    train_parser.add_argument("--output-dir", default="outputs/scratch")
    train_parser.add_argument("--dataset", default="PolyAI/banking77")
    train_parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--label-smoothing", type=float, default=0.0)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--validation-ratio", type=float, default=0.1)
    train_parser.add_argument("--confidence-threshold", type=float, default=0.65)
    train_parser.add_argument("--max-length", type=int, default=64)
    train_parser.add_argument("--max-vocabulary-size", type=int, default=20_000)
    train_parser.add_argument("--min-frequency", type=int, default=1)
    train_parser.add_argument("--d-model", type=int, default=128)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--feedforward-size", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(handler=train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Re-evaluate a saved model")
    evaluate_parser.add_argument("--experiment-dir", required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.set_defaults(handler=evaluate)

    predict_parser = subparsers.add_parser("predict", help="Classify one or more messages")
    predict_parser.add_argument("text", nargs="+")
    predict_parser.add_argument("--experiment-dir", required=True)
    predict_parser.add_argument("--top-k", type=int, default=3)
    predict_parser.add_argument("--confidence-threshold", type=float)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.set_defaults(handler=predict)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
