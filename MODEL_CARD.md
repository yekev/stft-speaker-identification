# Model Card: RAVDESS ResNet-18 Speaker Identifier

## Model summary

This model performs 24-class, closed-set speaker identification on the
audio-only speech subset of RAVDESS. It converts a three-second waveform into a
log-power STFT spectrogram and uses a one-channel ResNet-18 classifier to
predict an enrolled actor ID.

| Property | Value |
|---|---|
| Architecture | ResNet-18, trained from scratch |
| Classes | 24 RAVDESS actors |
| Input | Mono WAV, resampled to 16 kHz, three seconds |
| Feature | Log-power STFT |
| STFT configuration | FFT 512, window 400, hop 160 |
| Parameters | Approximately 11.18 million |
| Test accuracy | 0.8056 (116 / 144) |
| Weight file size | 44,801,668 bytes |

## Intended use

The model is intended for education, reproducibility, and demonstration of a
classic signal-processing plus deep-learning pipeline. It may be used to
reproduce the reported experiment or to study closed-set classification.

It is **not** intended for:

- identifying people outside the 24 RAVDESS actors;
- authentication, surveillance, access control, or forensic use;
- inferring identity from arbitrary real-world recordings;
- commercial use without an appropriate RAVDESS license.

The classifier always returns one of its 24 known classes, even when the input
speaker is unknown.

## Training and evaluation data

The model was trained on the 1,440-file RAVDESS audio-only speech subset using
a deterministic per-actor split:

| Split | Samples |
|---|---:|
| Train | 1,152 |
| Validation | 144 |
| Test | 144 |

RAVDESS contains acted speech from 24 professional actors speaking two fixed
English statements. It is not representative of unconstrained speakers,
languages, microphones, environments, or speaking styles.

## Preprocessing

1. Convert to mono.
2. Resample from 48 kHz to 16 kHz.
3. Normalize by peak amplitude.
4. Randomly crop training recordings or center-crop evaluation recordings to
   three seconds; zero-pad shorter recordings.
5. Compute a Hann-window log-power STFT.

## Evaluation

The checkpoint was re-evaluated with the portfolio implementation on all 144
held-out test recordings. The exact result was 116 correct predictions and an
accuracy of `0.805555...`. Per-class metrics and the confusion matrix are in
[`results/baseline/`](results/baseline/).

## Limitations and risks

- This is a closed-set experiment: all actor identities appear during training.
- The dataset is small, acted, English-only, and recorded under controlled
  conditions.
- Per-class test support is only six recordings, so class-level metrics have
  high variance.
- Identity predictions can be wrong and should not be used to make decisions
  about people.
- The checkpoint does not implement open-set rejection or uncertainty
  calibration.

## Weight download and integrity

Release asset:

`ravdess-speaker-id-resnet18-hop160.pt`

SHA-256:

```text
c8d3a5647a16d381bb2439b6492a240f8735937dbc6918a593a90d2274aacb31
```

The loader uses tensor-only checkpoint loading when supported by PyTorch. As
with any downloaded binary model, verify the checksum before use.

## License and attribution

The checkpoint is distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) because
it was trained on RAVDESS. The dataset is not included in the model release.

Please cite the dataset:

> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database
> of Emotional Speech and Song (RAVDESS). *PLOS ONE, 13*(5), e0196391.

- Official dataset: https://zenodo.org/records/1188976
- Paper: https://doi.org/10.1371/journal.pone.0196391
