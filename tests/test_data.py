from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from speaker_id.config import AudioConfig
from speaker_id.data import (
    create_stratified_splits,
    load_waveform,
    parse_ravdess_filename,
)


class DataTests(unittest.TestCase):
    def test_parse_ravdess_filename(self) -> None:
        metadata = parse_ravdess_filename("03-01-06-02-01-02-24.wav")
        self.assertEqual(metadata["actor"], "24")
        self.assertEqual(metadata["emotion_name"], "fearful")

    def test_rejects_non_speech_subset(self) -> None:
        with self.assertRaises(ValueError):
            parse_ravdess_filename("03-02-06-02-01-02-24.wav")

    def test_split_is_portable_and_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_path = Path(directory)
            root = temporary_path / "ravdess"
            for actor in ("01", "02"):
                actor_dir = root / f"Actor_{actor}"
                actor_dir.mkdir(parents=True)
                for repetition in range(1, 6):
                    (actor_dir / f"03-01-03-01-01-{repetition:02d}-{actor}.wav").touch()

            output = temporary_path / "split.json"
            result = create_stratified_splits(
                root,
                output,
                seed=42,
                validation_ratio=0.2,
                test_ratio=0.2,
            )

            self.assertNotIn("dataset_root", result)
            self.assertEqual(len(result["train"]), 6)
            self.assertEqual(len(result["val"]), 2)
            self.assertEqual(len(result["test"]), 2)
            self.assertTrue(all("\\" not in item["path"] for item in result["train"]))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_audio_has_fixed_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            samples = np.linspace(-0.25, 0.25, 4_000, dtype=np.float32)
            sf.write(path, samples, 8_000)

            waveform = load_waveform(
                path,
                AudioConfig(sample_rate=16_000, clip_seconds=1.0),
            )
            self.assertEqual(waveform.shape, (1, 16_000))
            self.assertLessEqual(float(waveform.abs().max()), 1.0)
