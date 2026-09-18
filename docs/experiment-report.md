# Experiment report

## Objective

The task is 24-way closed-set speaker identification on the audio-only speech
subset of RAVDESS. Each recording is mapped to the actor ID encoded in its
filename.

## Dataset

RAVDESS contributes 1,440 recordings: 60 samples for each of 24 actors. A
seed-42 per-actor split is used:

| Split | Samples | Samples per actor |
|---|---:|---:|
| Train | 1,152 | 48 |
| Validation | 144 | 6 |
| Test | 144 | 6 |

All split paths are relative, making the manifest portable across machines.

## Signal pipeline

1. Read a 48 kHz WAV recording and convert it to mono.
2. Resample to 16 kHz.
3. Peak-normalize the waveform.
4. Randomly crop training clips or center-crop evaluation clips to three
   seconds; zero-pad shorter recordings.
5. Apply a Hann-window STFT and convert magnitude to log power.
6. Feed the single-channel spectrogram to a ResNet-18 classifier.

The baseline parameters are `n_fft=512`, `win_length=400`, and
`hop_length=160`. At 16 kHz, these correspond to a 25 ms analysis window and a
10 ms step.

## Training setup

| Item | Value |
|---|---|
| Model | ResNet-18, trained from scratch |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 10 |
| Checkpoint selection | Highest validation accuracy |
| Random seed | 42 |

## Results

| Experiment | Hop length | Best validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Baseline | 160 | 0.7938 | **0.8056** |
| Ablation | 320 | 0.7625 | 0.7569 |

Random chance is approximately 0.0417. The baseline reaches 0.8056, while the
larger-hop ablation loses 0.0487 absolute test accuracy.

## Interpretation

Doubling the hop length halves the approximate number of time steps. The more
compact representation is faster, but it discards short-time detail associated
with pitch movement, formant transitions, articulation, and speaking rhythm.
The observed result supports the 10 ms baseline hop for this dataset.

## Reproducibility check

The saved baseline checkpoint was re-evaluated on all 144 held-out samples in
the original Python 3.10 / PyTorch 2.1.2 CPU environment. It reproduced the
reported test accuracy exactly: `116 / 144 = 0.805555...`.

## Limitations

The experiment uses a small acted dataset, fixed phrases, and the same enrolled
speakers in all splits. It measures closed-set classification under controlled
conditions and is not evidence of open-set speaker recognition or production
biometric reliability.
