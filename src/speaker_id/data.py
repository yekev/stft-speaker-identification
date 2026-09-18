from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from torch.nn import functional as neural_functional
from torch.utils.data import Dataset

from .config import AudioConfig


EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_filename(filename: str | Path) -> dict[str, str]:
    """Parse the seven fields encoded in a RAVDESS filename."""

    fields = Path(filename).stem.split("-")
    if len(fields) != 7:
        raise ValueError(f"Invalid RAVDESS filename: {filename}")

    modality, channel, emotion, intensity, statement, repetition, actor = fields
    if modality != "03" or channel != "01":
        raise ValueError(
            "This project expects the audio-only speech subset "
            f"(modality=03, channel=01), received {filename}."
        )
    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion code in {filename}: {emotion}")
    if not actor.isdigit() or not 1 <= int(actor) <= 24:
        raise ValueError(f"Invalid actor code in {filename}: {actor}")

    return {
        "modality": modality,
        "channel": channel,
        "emotion": emotion,
        "emotion_name": EMOTIONS[emotion],
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
    }


def find_actor_root(dataset_root: str | Path) -> Path:
    """Return the directory directly containing ``Actor_01`` ... folders."""

    root = Path(dataset_root).expanduser().resolve()
    candidates = (root, root / "audio_speech_actors_01-24")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("Actor_*")):
            return candidate
    raise FileNotFoundError(
        "Could not find RAVDESS Actor_* directories under "
        f"{root}. See the dataset instructions in README.md."
    )


def list_audio_files(dataset_root: str | Path) -> list[Path]:
    """List all RAVDESS WAV files in deterministic order."""

    return sorted(find_actor_root(dataset_root).glob("Actor_*/*.wav"))


def _records(paths: Iterable[Path], actor_root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(actor_root).as_posix(),
            "actor": parse_ravdess_filename(path.name)["actor"],
        }
        for path in paths
    ]


def create_stratified_splits(
    dataset_root: str | Path,
    output_file: str | Path,
    *,
    seed: int = 42,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    """Create deterministic, per-speaker train/validation/test splits."""

    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and sum to less than 1.")

    actor_root = find_actor_root(dataset_root)
    actor_dirs = sorted(path for path in actor_root.glob("Actor_*") if path.is_dir())
    if not actor_dirs:
        raise ValueError(f"No actor directories found in {actor_root}")

    rng = random.Random(seed)
    result: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "actors": [],
        "train": [],
        "val": [],
        "test": [],
    }

    for actor_dir in actor_dirs:
        actor = actor_dir.name.removeprefix("Actor_")
        files = sorted(actor_dir.glob("*.wav"))
        if len(files) < 3:
            raise ValueError(f"Actor {actor} needs at least three recordings.")
        rng.shuffle(files)

        test_count = max(1, int(len(files) * test_ratio))
        validation_count = max(1, int(len(files) * validation_ratio))
        train_end = len(files) - validation_count - test_count

        result["actors"].append(actor)
        result["train"].extend(_records(files[:train_end], actor_root))
        result["val"].extend(
            _records(files[train_end : train_end + validation_count], actor_root)
        )
        result["test"].extend(_records(files[train_end + validation_count :], actor_root))

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_waveform(
    path: str | Path,
    config: AudioConfig,
    *,
    random_crop: bool = False,
) -> torch.Tensor:
    """Load, resample, normalize, and crop/pad one audio file."""

    samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    waveform = waveform.mean(dim=0, keepdim=True)

    if source_rate != config.sample_rate:
        waveform = audio_functional.resample(waveform, source_rate, config.sample_rate)

    if config.normalize:
        peak = waveform.abs().amax().clamp_min(1e-8)
        waveform = waveform / peak

    target = config.sample_count
    length = waveform.shape[-1]
    if length > target:
        maximum_start = length - target
        start = (
            int(torch.randint(maximum_start + 1, (1,)).item())
            if random_crop
            else maximum_start // 2
        )
        waveform = waveform[:, start : start + target]
    elif length < target:
        waveform = neural_functional.pad(waveform, (0, target - length))

    return waveform.contiguous()


class RAVDESSpeakerDataset(Dataset[tuple[torch.Tensor, int, dict[str, str]]]):
    """RAVDESS audio-only speech dataset for closed-set speaker ID."""

    def __init__(
        self,
        dataset_root: str | Path,
        split_file: str | Path,
        split: str,
        audio_config: AudioConfig,
        *,
        training: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.actor_root = find_actor_root(dataset_root)
        split_data = json.loads(Path(split_file).read_text(encoding="utf-8"))
        self.items: list[dict[str, str]] = split_data[split]
        self.actors: list[str] = sorted(split_data["actors"])
        self.actor_to_index = {actor: index for index, actor in enumerate(self.actors)}
        self.audio_config = audio_config
        self.training = training

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        item = self.items[index]
        audio_path = self.actor_root / Path(item["path"])
        waveform = load_waveform(
            audio_path,
            self.audio_config,
            random_crop=self.training,
        )
        metadata = parse_ravdess_filename(audio_path.name)
        return waveform, self.actor_to_index[item["actor"]], metadata
