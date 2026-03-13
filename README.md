# MALTO — Human vs. AI-Generated Text Classification

A multi-class text classification pipeline built for the [MALTO Recruitment Hackathon](https://www.kaggle.com/competitions/malto-recruitment-hackathon) hosted by [MALTO](https://malto.ai) and [Politecnico di Torino](https://www.polito.it/).

The goal is to classify text as human-written or identify which of five AI models generated it (DeepSeek, Grok, Claude, Gemini, ChatGPT). 

* **Task:** 6-class text classification with severe class imbalance (19:1)
* **Metric:** Macro F1 Score
* **Best Public LB:** 0.92170
* **Best OOF CV:** 0.9591

---

## Dataset Statistics

The dataset consists of 2,400 training samples and 600 test samples. The primary challenge is the confusion pair between the two minority classes (DeepSeek and Grok), which act as the main bottleneck for the Macro F1 metric.

| Class | Label | Train Samples | Share |
|---|---|---|---|
| Human | 0 | 1,520 | 63.3% |
| ChatGPT | 5 | 320 | 13.3% |
| Gemini | 4 | 240 | 10.0% |
| Grok | 2 | 160 | 6.7% |
| DeepSeek | 1 | 80 | 3.3% |
| Claude | 3 | 80 | 3.3% |

---

## Pipeline Overview

The solution follows an 8-phase reproducible pipeline. Run notebooks sequentially (**Phase 1 > 2 > 3 > 4 > 5 > 6**). Each phase saves artifacts into the `models/` directory that the next phase consumes. Phases 7 and 8 build on top of artifacts saved from Phases 2 and 6.

| Phase | Notebook | Description | Best OOF F1 |
|---|---|---|---|
| 1 | `eda_and_baseline.ipynb` | EDA, class distribution, text statistics, TF-IDF + LR/LinearSVC baselines | 0.9305 |
| 2 | `feature_engineering.ipynb` | 46 stylometric features, 100k TF-IDF dims, 10 classical models, 5-fold CV | 0.9194 |
| 3 | `transformer_finetuning.ipynb` | DeBERTa-v3-base with Focal Loss, LLRD, cosine scheduler, temperature scaling | 0.9151 |
| 4 | `ensemble_optimization.ipynb` | Weighted soft voting, threshold optimization, pseudo-labeling | 0.9552 |
| 5 | `advanced_ensemble.ipynb` | Stacking meta-learner, rank averaging, per-class specialist voting, confidence-aware blending | 0.9552 |
| 6 | `final_submission.ipynb` | Score dashboard, submission agreement analysis, top-2 selection, reproducibility snapshot | N/A |
| 7 | `deberta_kfold_training.ipynb` | DeBERTa 5-fold CV, weighted voting with SVC + LR | 0.9500 |
| 8 | `ensemble_boosting.ipynb` | XGBoost + LightGBM diversity, 5-model ensemble, threshold tuning | 0.9591 |

---

## Approach & Key Techniques

### Models & Features
* **Feature Engineering:** 46 handcrafted linguistic and stylometric features (readability, character entropy, AI-phrasing, punctuation patterns) combined with 100k TF-IDF sparse features (50k word + 50k character n-grams).
* **Classical ML:** LinearSVC (C=5.0) and Logistic Regression (C=2.0) trained with balanced class weights and 5-fold stratified CV.
* **Transformer:** DeBERTa-v3-base fine-tuned with Focal Loss (gamma=2.0) to handle the 19:1 class imbalance, layer-wise learning rate decay (factor=0.9), and post-hoc temperature scaling on a held-out calibration set. Pre-tokenization is used to avoid bottlenecks.
* **Tree-Based:** XGBoost and LightGBM trained on the same feature set to add ensemble diversity.

### Ensemble Strategy
The final prediction blends five base models through weighted soft voting. A grid search over out-of-fold predictions identified the optimal weights. We then applied greedy per-class threshold multipliers to boost recall on minority classes (DeepSeek and Grok represent only 10% of training data combined), and used high-confidence test predictions for pseudo-labeling to augment the training set.

---

## Results

### Score Progression

| Submission | Method | Public LB |
|---|---|---|
| `01_tfidf_svc_baseline` | TF-IDF + LinearSVC | 0.84123 |
| `05_final_stacking` | Stacking meta-LR (DeBERTa + SVC + LR) | 0.90974 |
| `06_deberta_5fold` | DeBERTa 5-fold only | 0.91648 |
| `07_weighted_vote_best` | Weighted vote (DeBERTa 0.65 + SVC 0.25 + LR 0.10) | **0.92170** |

### Per-Class OOF Performance (Best Ensemble)

| Class | F1 | Support |
|---|---|---|
| Human | 0.998 | 1,520 |
| ChatGPT | 0.997 | 320 |
| Gemini | 0.998 | 240 |
| Grok | 0.882 | 160 |
| DeepSeek | 0.791 | 80 |
| Claude | 0.988 | 80 |

---

## Project Structure

```text
MALTO/
├── src/
│   ├── __init__.py          # Public API re-exports
│   ├── features.py          # extract_features() -- 46 handcrafted features
│   ├── models.py            # FocalLoss, TemperatureScaler, ensemble utilities
│   └── utils.py             # Constants, data loading, submission helpers
├── notebooks/               # Jupyter notebooks for Phases 1-8
├── figures/                 # Visualization PNGs
├── models/                  # Saved model configs (JSON tracked, binaries git-ignored)
├── checkpoints/             # DeBERTa model checkpoints (git-ignored)
├── submissions/             # Kaggle submission CSVs
├── competition_info.json    # Competition metadata
├── PROJECT_PLAN.md          # Full sprint plan
├── requirements.txt
├── LICENSE                  
└── README.md
