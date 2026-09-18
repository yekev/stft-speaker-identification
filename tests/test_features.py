import unittest

import torch

from speaker_id.config import FeatureConfig
from speaker_id.features import LogSTFT


class FeatureTests(unittest.TestCase):
    def test_log_stft_shape_and_finiteness(self) -> None:
        waveform = torch.zeros(2, 1, 48_000)
        spectrogram = LogSTFT(FeatureConfig())(waveform)
        self.assertEqual(spectrogram.shape, (2, 1, 257, 301))
        self.assertTrue(bool(torch.isfinite(spectrogram).all()))
