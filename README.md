# MALTO — 2nd Place Solution

**2nd place** on the [MALTO Recruitment Hackathon](https://www.kaggle.com/competitions/malto-recruitment-hackathon) hosted by [MALTO](https://malto.ai) and [Politecnico di Torino](https://www.polito.it/).

| Metric | Score |
|---|---|
| **Public LB (Macro F1)** | **0.95341** |
| OOF CV (5-fold) | 0.9575 ± 0.0044 |
| Blended OOF F1 | 0.9605 |

![Competition Results](figures/competition_results.png)

---

## Task

Classify text as human-written or identify which AI model generated it across 6 classes:

| Class | Train Samples | Share |
|---|---|---|
| Human | 1,520 | 63.3% |
| ChatGPT | 320 | 13.3% |
| Gemini | 240 | 10.0% |
| Grok | 160 | 6.7% |
| DeepSeek | 80 | 3.3% |
| Claude | 80 | 3.3% |

The main challenge is severe class imbalance (19:1 ratio) with DeepSeek and Grok as the hardest minority classes.

---

## Solution

The solution ensembles a fine-tuned transformer with a classical n-gram model, optimised via Nelder-Mead on out-of-fold predictions.

### Pipeline

```
ModernBERT-base (5-fold CV) ─┬─ Temperature Scaling ─┬─ Nelder-Mead ─── Threshold ─── Submission
                              │                       │   Blend (70/30)   Nudge
Full-data ModernBERT (7 ep) ──┘                       │
                                                      │
TF-IDF + Calibrated SVC (5-fold CV) ──────────────────┘
```

### Key Techniques

| Component | Details |
|---|---|
| **Transformer** | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) fine-tuned with LDAM loss, gradual DRW (10x cap), label smoothing (ε=0.1) |
| **Optimizer** | AdamW with layer-wise learning rate decay (LLRD=0.9), cosine schedule, 10% warmup |
| **Classical Model** | TF-IDF (50k char 3-5 grams + 50k word 1-2 grams) → Calibrated LinearSVC (C=5.0) |
| **Ensemble** | Nelder-Mead optimisation over 6 initialisations on OOF predictions |
| **Full-data Model** | Trained on all 2,400 samples (7 epochs, LR×0.8), blended with fold-average at α=0.6 |
| **Post-processing** | Temperature scaling (T=0.30) + conservative per-class threshold nudge [0.85, 1.20] |
| **Training** | Kaggle T4×2 GPUs via DataParallel, ~155 min total |

### Per-Class OOF Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Human | 1.00 | 1.00 | 1.00 |
| DeepSeek | 0.85 | 0.82 | 0.84 |
| Grok | 0.92 | 0.92 | 0.92 |
| Claude | 1.00 | 1.00 | 1.00 |
| Gemini | 0.99 | 1.00 | 0.99 |
| ChatGPT | 1.00 | 1.00 | 1.00 |

### Score Progression

| Submission | Method | Public LB |
|---|---|---|
| TF-IDF + LinearSVC baseline | Classical only | 0.84123 |
| DeBERTa 5-fold | Transformer only | 0.91648 |
| Weighted vote (DeBERTa + SVC + LR) | Multi-model ensemble | 0.92170 |
| ModernBERT + LDAM + DRW (3-fold) | Single transformer | 0.94120 |
| **ModernBERT + SVC ensemble (5-fold)** | **Final solution** | **0.95341** |

---

## Repository Structure

```
MALTO/
├── notebooks/
│   ├── solution.ipynb          # Full pipeline (main notebook)
│   └── solution_v9_tpu.ipynb   # TPU variant with per-class Nelder-Mead ensemble
├── malto_model/
│   └── ensemble_config.json    # Saved ensemble parameters
├── submission.csv              # Base submission (0.95341 public F1)
├── submission_final.csv        # Final 2nd-place submission
├── src/                        # Utility modules (features, models, utils)
├── archive/                    # Previous experiment notebooks and artifacts
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

Upload `notebooks/solution.ipynb` to Kaggle with GPU T4×2 enabled, attach the competition dataset, and run all cells.

```python
# Or load the saved model locally:
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("malto_model")
tokenizer = AutoTokenizer.from_pretrained("malto_model")
```

## Requirements

```
torch>=2.0
transformers>=4.40
scikit-learn>=1.3
scipy
numpy
pandas
tqdm
```

---

## License

MIT — see [LICENSE](LICENSE).
