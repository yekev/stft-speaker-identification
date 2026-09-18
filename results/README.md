# Verified results

These artifacts were regenerated from the two saved coursework checkpoints
using the refactored evaluation pipeline and the portable seed-42 split.

| Directory | Configuration | Correct | Accuracy |
|---|---|---:|---:|
| `baseline/` | `hop_length=160` | 116 / 144 | 0.8056 |
| `hop320/` | `hop_length=320` | 109 / 144 | 0.7569 |

Each directory contains:

- `metrics.json`: sample count, accuracy, per-class metrics, and confusion matrix;
- `classification_report.txt`: human-readable precision/recall/F1 report;
- `confusion_matrix.png`: independently regenerated visualization.

The model checkpoints are not committed to Git. They are approximately 42.7
MiB each and should be distributed separately as release artifacts if needed.
