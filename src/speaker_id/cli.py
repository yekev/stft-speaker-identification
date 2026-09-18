from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AudioConfig, FeatureConfig, TrainingConfig
from .data import create_stratified_splits
from .evaluation import evaluate, save_evaluation
from .inference import predict_file
from .training import train


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic RAVDESS splits.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ravdess"))
    parser.add_argument("--output", type=Path, default=Path("splits/split_seed42.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = create_stratified_splits(args.dataset_root, args.output, seed=args.seed)
    counts = {name: len(result[name]) for name in ("train", "val", "test")}
    print(json.dumps({"split_file": str(args.output), "counts": counts}, indent=2))


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train a speaker-ID classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ravdess"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/split_seed42.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--model", choices=("resnet18", "vgg11_bn"), default="resnet18")
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainingConfig(
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        output_dir=args.output_dir,
        features=FeatureConfig(hop_length=args.hop_length),
        model_name=args.model,
        seed=args.seed,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        device=args.device,
    )
    summary = train(config)
    print(json.dumps(summary, indent=2, default=str))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ravdess"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/split_seed42.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = evaluate(
        args.dataset_root,
        args.split_file,
        args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        device_preference=args.device,
    )
    save_evaluation(result, args.output_dir)
    print(result["classification_report_text"])
    print(f"Accuracy: {result['accuracy']:.4f} ({result['sample_count']} samples)")


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-set inference on one WAV file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = predict_file(
        args.audio,
        args.checkpoint,
        top_k=args.top_k,
        device_preference=args.device,
    )
    print(json.dumps(result, indent=2))
