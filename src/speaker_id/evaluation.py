from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .checkpoint import (
    actors_from_checkpoint,
    audio_config_from_checkpoint,
    feature_config_from_checkpoint,
)
from .data import RAVDESSpeakerDataset
from .features import LogSTFT
from .models import create_classifier
from .utils import load_checkpoint, resolve_device


@torch.no_grad()
def evaluate(
    dataset_root: str | Path,
    split_file: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    batch_size: int = 32,
    workers: int = 0,
    device_preference: str = "cuda",
) -> dict[str, Any]:
    """Evaluate a checkpoint and return exact sample-level metrics."""

    device = resolve_device(device_preference)
    checkpoint = load_checkpoint(checkpoint_path, device)
    audio_config = audio_config_from_checkpoint(checkpoint)
    feature_config = feature_config_from_checkpoint(checkpoint)
    checkpoint_actors = actors_from_checkpoint(checkpoint)

    dataset = RAVDESSpeakerDataset(
        dataset_root,
        split_file,
        split,
        audio_config,
    )
    if dataset.actors != checkpoint_actors:
        raise ValueError("Dataset actor ordering does not match the checkpoint.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    feature_extractor = LogSTFT(feature_config).to(device).eval()
    model = create_classifier(
        str(checkpoint["model_name"]),
        len(checkpoint_actors),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    expected: list[int] = []
    predicted: list[int] = []
    for waveforms, targets, _ in loader:
        logits = model(feature_extractor(waveforms.to(device)))
        expected.extend(targets.tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())

    labels = list(range(len(checkpoint_actors)))
    matrix = confusion_matrix(expected, predicted, labels=labels)
    report_text = classification_report(
        expected,
        predicted,
        labels=labels,
        target_names=[f"Actor_{actor}" for actor in checkpoint_actors],
        digits=4,
        zero_division=0,
    )
    report_data = classification_report(
        expected,
        predicted,
        labels=labels,
        target_names=[f"Actor_{actor}" for actor in checkpoint_actors],
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(np.mean(np.asarray(expected) == np.asarray(predicted)))
    return {
        "split": split,
        "sample_count": len(expected),
        "accuracy": accuracy,
        "actors": checkpoint_actors,
        "confusion_matrix": matrix,
        "classification_report": report_data,
        "classification_report_text": report_text,
    }


def save_evaluation(result: dict[str, Any], output_dir: str | Path) -> None:
    """Save machine-readable metrics, a text report, and a confusion matrix."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    serializable = {
        key: value
        for key, value in result.items()
        if key not in {"confusion_matrix", "classification_report_text"}
    }
    serializable["confusion_matrix"] = result["confusion_matrix"].tolist()
    (destination / "metrics.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )
    (destination / "classification_report.txt").write_text(
        result["classification_report_text"],
        encoding="utf-8",
    )

    matrix = result["confusion_matrix"]
    actors = result["actors"]
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, cmap="viridis")
    axis.set(
        title=f"Confusion matrix — accuracy {result['accuracy']:.4f}",
        xlabel="Predicted speaker",
        ylabel="True speaker",
        xticks=range(len(actors)),
        yticks=range(len(actors)),
        xticklabels=actors,
        yticklabels=actors,
    )
    axis.tick_params(axis="x", labelrotation=90)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrix.png", dpi=180)
    plt.close(figure)
