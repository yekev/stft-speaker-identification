from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .data import RAVDESSpeakerDataset
from .features import LogSTFT
from .models import create_classifier
from .utils import resolve_device, seed_everything, seed_worker


def _loader(
    dataset: RAVDESSpeakerDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if workers else None,
        generator=generator,
    )


def _run_epoch(
    model: torch.nn.Module,
    feature_extractor: LogSTFT,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    description: str,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for waveforms, targets, _ in tqdm(loader, desc=description):
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.no_grad():
                features = feature_extractor(waveforms)
            logits = model(features)
            loss = criterion(logits, targets)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = targets.shape[0]
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())

    return total_loss / total_samples, total_correct / total_samples


def train(config: TrainingConfig) -> dict[str, Any]:
    """Train a spectrogram classifier and save the best validation checkpoint."""

    seed_everything(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RAVDESSpeakerDataset(
        config.dataset_root,
        config.split_file,
        "train",
        config.audio,
        training=True,
    )
    validation_dataset = RAVDESSpeakerDataset(
        config.dataset_root,
        config.split_file,
        "val",
        config.audio,
    )
    pin_memory = device.type == "cuda"
    train_loader = _loader(
        train_dataset,
        batch_size=config.batch_size,
        workers=config.workers,
        shuffle=True,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=config.batch_size,
        workers=config.workers,
        shuffle=False,
        seed=config.seed,
        pin_memory=pin_memory,
    )

    feature_extractor = LogSTFT(config.features).to(device).eval()
    model = create_classifier(
        config.model_name,
        len(train_dataset.actors),
        pretrained=config.pretrained,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_accuracy = -1.0
    best_path = config.output_dir / "best.pt"

    print(f"Device: {device}")
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model,
            feature_extractor,
            train_loader,
            criterion,
            device,
            optimizer,
            f"Epoch {epoch}/{config.epochs} [train]",
        )
        validation_loss, validation_accuracy = _run_epoch(
            model,
            feature_extractor,
            validation_loader,
            criterion,
            device,
            None,
            f"Epoch {epoch}/{config.epochs} [validation]",
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "schema_version": 1,
                    "model_name": config.model_name,
                    "num_classes": len(train_dataset.actors),
                    "state_dict": model.state_dict(),
                    "audio_config": config.audio.to_dict(),
                    "feature_config": config.features.to_dict(),
                    "actors": train_dataset.actors,
                    "history": history,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")

    summary = {
        "best_validation_accuracy": best_accuracy,
        "best_checkpoint": str(best_path),
        "history": history,
        "configuration": {
            **asdict(config),
            "dataset_root": str(config.dataset_root),
            "split_file": str(config.split_file),
            "output_dir": str(config.output_dir),
        },
    }
    (config.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
