import unittest

from speaker_id.checkpoint import (
    actors_from_checkpoint,
    audio_config_from_checkpoint,
    feature_config_from_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def test_legacy_checkpoint_metadata_is_supported(self) -> None:
        checkpoint = {
            "num_classes": 2,
            "audio_cfg": {"target_sr": 16_000, "clip_seconds": 3.0, "normalize": True},
            "feature": {"n_fft": 512, "hop_length": 160, "win_length": 400},
            "actor_to_index": {"02": 1, "01": 0},
        }
        self.assertEqual(audio_config_from_checkpoint(checkpoint).sample_rate, 16_000)
        self.assertEqual(feature_config_from_checkpoint(checkpoint).hop_length, 160)
        self.assertEqual(actors_from_checkpoint(checkpoint), ["01", "02"])
