from __future__ import annotations

from typing import Any

from .config import AudioConfig, FeatureConfig


def audio_config_from_checkpoint(checkpoint: dict[str, Any]) -> AudioConfig:
    values = checkpoint.get("audio_config", checkpoint.get("audio_cfg", {}))
    return AudioConfig(
        sample_rate=int(values.get("sample_rate", values.get("target_sr", 16_000))),
        clip_seconds=float(values.get("clip_seconds", 3.0)),
        normalize=bool(values.get("normalize", True)),
    )


def feature_config_from_checkpoint(checkpoint: dict[str, Any]) -> FeatureConfig:
    values = checkpoint.get("feature_config", checkpoint.get("feature", {}))
    return FeatureConfig(
        n_fft=int(values.get("n_fft", 512)),
        hop_length=int(values.get("hop_length", 160)),
        win_length=int(values.get("win_length", 400)),
        power=float(values.get("power", 2.0)),
        log_epsilon=float(values.get("log_epsilon", values.get("log_eps", 1e-10))),
        center=bool(values.get("center", True)),
    )


def actors_from_checkpoint(checkpoint: dict[str, Any]) -> list[str]:
    if "actors" in checkpoint:
        return list(checkpoint["actors"])
    mapping = checkpoint.get("actor_to_index")
    if not mapping:
        class_count = int(checkpoint["num_classes"])
        return [f"{index + 1:02d}" for index in range(class_count)]
    return [actor for actor, _ in sorted(mapping.items(), key=lambda item: item[1])]
