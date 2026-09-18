from __future__ import annotations

import torch

from .config import FeatureConfig


class LogSTFT(torch.nn.Module):
    """Convert batched waveforms into single-channel log-power spectrograms."""

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "window",
            torch.hann_window(config.win_length),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        if waveform.ndim != 2:
            raise ValueError(
                "Expected waveform shape [batch, time] or [batch, 1, time], "
                f"received {tuple(waveform.shape)}."
            )

        spectrum = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=self.config.center,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        power = magnitude.pow(self.config.power)
        return torch.log(power + self.config.log_epsilon).unsqueeze(1)
