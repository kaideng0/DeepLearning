"""Command-line entry point for training, evaluation, and prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from terraclass.checkpoints import load_checkpoint, save_checkpoint
from terraclass.data import build_transforms, make_dataloaders
from terraclass.engine import run_epoch
from terraclass.gradcam import create_gradcam
from terraclass.models import MODEL_NAMES, build_model, default_image_size, gradcam_target_layer
from terraclass.plots import plot_confusion_matrix, plot_history, save_gradcam_overlay
from terraclass.utils import resolve_device, save_json, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terraclass",
        description="Train and inspect PyTorch land-use classifiers on EuroSAT.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="train a custom CNN or ResNet18",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument("--model", choices=MODEL_NAMES, default="cnn")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--output-dir", default="outputs/experiment")
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--image-size", type=int)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--val-ratio", type=float, default=0.15)
    train_parser.add_argument("--test-ratio", type=float, default=0.15)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--freeze-backbone", action="store_true")
    train_parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--resume", help="resume from a last.pt checkpoint")
    train_parser.set_defaults(handler=train_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate a checkpoint on the held-out test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    evaluate_parser.add_argument("checkpoint")
    evaluate_parser.add_argument("--data-dir", default="data")
    evaluate_parser.add_argument("--output-dir", default="outputs/evaluation")
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    evaluate_parser.set_defaults(handler=evaluate_command)

    predict_parser = subparsers.add_parser(
        "predict",
        help="classify one image and optionally save a Grad-CAM overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict_parser.add_argument("checkpoint")
    predict_parser.add_argument("image")
    predict_parser.add_argument("--top-k", type=int, default=3)
    predict_parser.add_argument("--gradcam-output")
    predict_parser.add_argument("--device", default="auto")
    predict_parser.set_defaults(handler=predict_command)
    return parser


def train_command(args: argparse.Namespace) -> None:
    _validate_positive(args, "epochs", "batch_size", "learning_rate")
    if args.patience < 0:
        raise ValueError("patience cannot be negative")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    image_size = args.image_size or default_image_size(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and not (output_dir / "best.pt").exists():
        raise ValueError(
            "When resuming, --output-dir must be the original run directory containing best.pt."
        )
    print(f"Using device: {device}")

    data = make_dataloaders(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        download=args.download,
    )
    model = build_model(
        args.model,
        len(data.class_names),
        pretrained=args.pretrained and args.resume is None,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    start_epoch = 1
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        _verify_checkpoint(checkpoint, args.model, data.class_names, image_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_f1 = float(checkpoint.get("best_val_f1", -1.0))
        history = list(checkpoint.get("history", []))
        print(f"Resuming after epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            data.train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(model, data.val_loader, criterion, device)
        scheduler.step(float(val_metrics["macro_f1"]))
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        val_f1 = float(val_metrics["macro_f1"])
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        state = _checkpoint_state(
            args,
            model,
            optimizer,
            scheduler,
            epoch,
            image_size,
            data.class_names,
            best_val_f1,
            history,
        )
        save_checkpoint(state, output_dir / "last.pt")
        if improved:
            save_checkpoint(state, output_dir / "best.pt")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, F1 {train_metrics['macro_f1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f}, F1 {val_f1:.3f}"
        )
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    if not history:
        raise ValueError("No epochs were run. Increase --epochs when resuming this checkpoint.")

    best_checkpoint = load_checkpoint(output_dir / "best.pt", device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, data.test_loader, criterion, device)
    report = _format_report(test_metrics, data.class_names)
    save_json({"history": history, "test": report}, output_dir / "metrics.json")
    plot_history(history, output_dir / "history.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        data.class_names,
        output_dir / "confusion_matrix.png",
    )
    print(
        f"Test accuracy: {test_metrics['accuracy']:.3f} | "
        f"macro F1: {test_metrics['macro_f1']:.3f}"
    )
    print(f"Artifacts saved to {output_dir.resolve()}")


def evaluate_command(args: argparse.Namespace) -> None:
    _validate_positive(args, "batch_size")
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint.get("training_config", {})
    data = make_dataloaders(
        args.data_dir,
        image_size=int(checkpoint["image_size"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=float(config.get("val_ratio", 0.15)),
        test_ratio=float(config.get("test_ratio", 0.15)),
        seed=int(config.get("seed", 42)),
        download=args.download,
    )
    _verify_checkpoint(
        checkpoint,
        str(checkpoint["model_name"]),
        data.class_names,
        int(checkpoint["image_size"]),
    )
    model = _model_from_checkpoint(checkpoint, device)
    metrics = run_epoch(model, data.test_loader, nn.CrossEntropyLoss(), device)
    report = _format_report(metrics, data.class_names)
    output_dir = Path(args.output_dir)
    save_json(report, output_dir / "metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        data.class_names,
        output_dir / "confusion_matrix.png",
    )
    print(json.dumps(report, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    _validate_positive(args, "top_k")
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = _model_from_checkpoint(checkpoint, device)
    class_names = list(checkpoint["class_names"])
    _, transform = build_transforms(int(checkpoint["image_size"]))
    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    if args.gradcam_output:
        target_layer = gradcam_target_layer(model, str(checkpoint["model_name"]))
        heatmap, logits, explained_class = create_gradcam(model, target_layer, input_tensor)
    else:
        with torch.no_grad():
            logits = model(input_tensor).cpu()
        explained_class = int(logits.argmax(dim=1).item())

    probabilities = logits.softmax(dim=1)[0]
    top_k = min(args.top_k, len(class_names))
    values, indices = probabilities.topk(top_k)
    predictions = [
        {"class": class_names[index], "probability": round(float(value), 6)}
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]
    print(json.dumps({"image": str(args.image), "predictions": predictions}, indent=2))

    if args.gradcam_output:
        title = f"Predicted: {class_names[explained_class]} ({probabilities[explained_class]:.1%})"
        save_gradcam_overlay(
            args.image,
            heatmap.numpy(),
            args.gradcam_output,
            title=title,
        )
        print(f"Grad-CAM overlay saved to {Path(args.gradcam_output).resolve()}")


def _model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    model = build_model(
        str(checkpoint["model_name"]),
        len(checkpoint["class_names"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def _checkpoint_state(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    image_size: int,
    class_names: list[str],
    best_val_f1: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"handler"} and isinstance(value, (str, int, float, bool, type(None)))
    }
    return {
        "format_version": 1,
        "model_name": args.model,
        "image_size": image_size,
        "class_names": class_names,
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_config": config,
        "history": history,
    }


def _verify_checkpoint(
    checkpoint: dict[str, Any],
    model_name: str,
    class_names: list[str],
    image_size: int,
) -> None:
    if checkpoint["model_name"] != model_name:
        raise ValueError(
            f"Checkpoint uses {checkpoint['model_name']}, but --model is {model_name}."
        )
    if list(checkpoint["class_names"]) != class_names:
        raise ValueError("Checkpoint classes do not match this dataset")
    if int(checkpoint["image_size"]) != image_size:
        raise ValueError("Checkpoint image size does not match this run")


def _format_report(metrics: dict[str, object], class_names: list[str]) -> dict[str, object]:
    report: dict[str, object] = {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class": {},
    }
    per_class = report["per_class"]
    assert isinstance(per_class, dict)
    for index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": metrics["per_class_precision"][index],
            "recall": metrics["per_class_recall"][index],
            "f1": metrics["per_class_f1"][index],
            "support": metrics["support"][index],
        }
    return report


def _validate_positive(args: argparse.Namespace, *names: str) -> None:
    for name in names:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be greater than zero")


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
