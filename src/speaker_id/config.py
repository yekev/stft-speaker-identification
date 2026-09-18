from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioConfig:
    """Waveform preprocessing parameters."""

    sample_rate: int = 16_000
    clip_seconds: float = 3.0
    normalize: bool = True

    @property
    def sample_count(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureConfig:
    """Log-power STFT parameters."""

    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    power: float = 2.0
    log_epsilon: float = 1e-10
    center: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """End-to-end training configuration."""

    dataset_root: Path = Path("data/ravdess")
    split_file: Path = Path("splits/split_seed42.json")
    output_dir: Path = Path("outputs/baseline")
    audio: AudioConfig = AudioConfig()
    features: FeatureConfig = FeatureConfig()
    model_name: str = "resnet18"
    pretrained: bool = False
    seed: int = 42
    batch_size: int = 32
    workers: int = 0
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
