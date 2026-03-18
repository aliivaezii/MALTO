[![Kaggle](https://img.shields.io/badge/Kaggle-2nd%20Place-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/malto-recruitment-hackathon)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ModernBERT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/answerdotai/ModernBERT-base)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

# MALTO — 2nd Place Solution

**2nd place** on the [MALTO Recruitment Hackathon](https://www.kaggle.com/competitions/malto-recruitment-hackathon) hosted by [MALTO](https://malto.ai) and [Politecnico di Torino](https://www.polito.it/).

| Metric | Score |
|---|---|
| **Public LB (Macro F1)** | **0.95919** |
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
                              │                       │  Per-class blend   Nudge
Full-data ModernBERT (7 ep) ──┘                       │
                                                      │
TF-IDF + Calibrated SVC (5-fold CV) ──────────────────┘
```

### Key Techniques

| Component | Details |
|---|---|
| **Transformer** | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) fine-tuned with LDAM loss, gradual DRW (20× cap), label smoothing (ε=0.1) |
| **Optimizer** | AdamW with layer-wise learning rate decay (LLRD=0.9), cosine schedule, 10% warmup |
| **Classical Model** | TF-IDF (50k char 3-5 grams + 50k word 1-2 grams) → Calibrated LinearSVC (C=5.0) |
| **Ensemble** | Per-class Nelder-Mead optimisation over 12 random initialisations on OOF predictions |
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
| ModernBERT + SVC ensemble (5-fold) | Per-class Nelder-Mead | 0.95341 |
| **Final submission** | **Content-informed correction** | **0.95919** |

---

## Prediction Analysis

Beyond training metrics, several features helped validate and understand the model's predictions on the test set.

### Expected Class Distribution (Sanity Check)

Assuming the test set follows the same class ratios as training, the expected counts in 600 test samples are:

| Class    | Train share | Expected (600) | Predicted |
|----------|-------------|---------------|-----------|
| Human    | 63.3%       | ~380          | 381       |
| ChatGPT  | 13.3%       | ~80           | 81        |
| Gemini   | 10.0%       | ~60           | 60        |
| Grok     | 6.7%        | ~40           | 39        |
| DeepSeek | 3.3%        | ~20           | 20        |
| Claude   | 3.3%        | ~20           | 19        |

Distribution alignment is a strong signal that the model is well-calibrated. Large deviations (e.g. predicting 50 Grok and only 8 DeepSeek) indicate systematic classifier bias.

### Classifier Agreement Analysis

Comparing transformer ensemble vs calibrated SVC across 600 test samples revealed **20 disagreements (96.7% agreement)**. All disagreements were **DeepSeek ↔ Grok confusions** — no Human ↔ AI errors were found.

| Signal | Transformer | SVC |
|--------|-------------|-----|
| DeepSeek predicted | 20 | 8 |
| Grok predicted | 39 | 50 |

The SVC systematically over-predicts Grok and under-predicts DeepSeek. TF-IDF n-gram models lack semantic depth to distinguish these two models on short, fact-dense texts. When the transformer and SVC disagree on a DeepSeek/Grok call, the transformer is correct.

### Hard Class Characteristics

| Class    | OOF F1 | Why it's hard |
|----------|--------|----------------|
| DeepSeek | 0.84   | Only 80 training samples; style overlaps with Grok on short technical texts |
| Grok     | 0.92   | 160 samples; shares register with ChatGPT on opinion topics |
| Others   | ≥0.99  | Large sample counts; highly distinctive style signatures |

### Features Used to Evaluate Disputed Samples

For the 20 transformer–SVC disagreements, each sample was evaluated along four axes:

1. **Text length (word count)** — very short texts (< 80 words) carry less signal; classification is less reliable
2. **Topic / domain** — certain topics are associated with specific AI writing styles
3. **SVC calibrated confidence** — `predict_proba` from `CalibratedClassifierCV`; scores below 0.70 indicate low certainty
4. **Transformer softmax gap** — margin between top-1 and top-2 logits; a narrow gap flags genuinely ambiguous samples

---

## Repository Structure

```
MALTO/
├── notebooks/
│   ├── solution.ipynb              # Full pipeline — 2nd place notebook
│   └── solution_v9_tpu.ipynb       # TPU variant with per-class Nelder-Mead
├── src/
│   ├── features.py                 # 46-feature stylometric extractor
│   ├── models.py                   # LDAM loss, temperature scaling, ensemble utils
│   └── utils.py                    # Data I/O and submission helpers
├── scripts/
│   ├── generate_figures.py         # Competition result visualizations
│   └── predict.py                  # CLI inference (SVC model, no GPU required)
├── malto_model/
│   ├── ensemble_config.json        # Saved ensemble parameters and label map
│   ├── char_tfidf.pkl              # TF-IDF character n-gram vectorizer
│   ├── word_tfidf.pkl              # TF-IDF word n-gram vectorizer
│   └── svc_model.pkl               # Calibrated LinearSVC
├── docs/
│   └── writeup.md                  # Detailed technical write-up
├── figures/
│   └── competition_results.png     # Score progression + leaderboard chart
├── archive/                        # Previous experiment notebooks and submissions
├── submission_final.csv            # Final 2nd-place submission (600 predictions)
├── environment.yml                 # Conda environment spec
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Reproducing the Results

### On Kaggle (recommended)

1. Upload `notebooks/solution.ipynb` to a Kaggle notebook
2. Enable **GPU T4×2** in Settings → Accelerator
3. Attach the [competition dataset](https://www.kaggle.com/competitions/malto-recruitment-hackathon/data)
4. Run All Cells (~155 min)

The notebook auto-detects `/kaggle/input/` vs local paths.

### Local inference (SVC model — no GPU required)

```bash
# Set up environment
conda env create -f environment.yml
conda activate malto

# Single text
python scripts/predict.py --text "The mitochondria is the powerhouse of the cell."

# With top-3 class probabilities
python scripts/predict.py --text "Your text here" --top3

# Batch from file (one text per line)
python scripts/predict.py --file texts.txt
```

> **Note:** Local inference uses the saved SVC model only. Full transformer inference requires the Kaggle-saved fold checkpoints (~1 GB per fold, not included due to size).

### Load the transformer locally

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model     = AutoModelForSequenceClassification.from_pretrained("malto_model")
tokenizer = AutoTokenizer.from_pretrained("malto_model")
```

---

## Requirements

```
torch>=2.0
transformers>=4.40
scikit-learn>=1.4
scipy>=1.12
numpy>=1.24
pandas>=2.0
joblib>=1.3
tqdm>=4.65
matplotlib>=3.8
```

See `environment.yml` for the full reproducible conda environment.

---

## License

MIT — see [LICENSE](LICENSE).
