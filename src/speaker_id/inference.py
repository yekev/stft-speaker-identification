from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    actors_from_checkpoint,
    audio_config_from_checkpoint,
    feature_config_from_checkpoint,
)
from .data import load_waveform
from .features import LogSTFT
from .models import create_classifier
from .utils import load_checkpoint, resolve_device


@torch.no_grad()
def predict_file(
    audio_path: str | Path,
    checkpoint_path: str | Path,
    *,
    top_k: int = 3,
    device_preference: str = "cuda",
) -> dict[str, Any]:
    """Predict one of the 24 enrolled RAVDESS actor IDs for a WAV file."""

    device = resolve_device(device_preference)
    checkpoint = load_checkpoint(checkpoint_path, device)
    actors = actors_from_checkpoint(checkpoint)
    waveform = load_waveform(
        audio_path,
        audio_config_from_checkpoint(checkpoint),
    ).unsqueeze(0).to(device)

    feature_extractor = LogSTFT(feature_config_from_checkpoint(checkpoint)).to(device).eval()
    model = create_classifier(str(checkpoint["model_name"]), len(actors)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    probabilities = model(feature_extractor(waveform)).softmax(dim=1)[0]
    count = min(max(1, top_k), len(actors))
    values, indices = probabilities.topk(count)
    predictions = [
        {
            "actor": f"Actor_{actors[index]}",
            "probability": float(value),
        }
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
    ]
    return {
        "audio": str(Path(audio_path)),
        "prediction": predictions[0]["actor"],
        "top_k": predictions,
        "scope": "Closed-set prediction over the 24 RAVDESS actors.",
    }
