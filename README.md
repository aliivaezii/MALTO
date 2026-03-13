# MALTO -- Human vs. AI-Generated Text Classification# Human vs. AI-Generated Text Classification



Multi-class text classification pipeline for theMulti-class text classification pipeline distinguishing human-written text from five AI generators (DeepSeek, Grok, Claude, Gemini, ChatGPT).

[MALTO Recruitment Hackathon](https://www.kaggle.com/competitions/malto-recruitment-hackathon)

on Kaggle. The task is to distinguish human-written text from five AI generators:**Metric:** Macro F1 Score  

DeepSeek, Grok, Claude, Gemini, and ChatGPT.**Task:** 6-class text classification with severe class imbalance (19:1)



**Best Public LB Score: 0.92170 (Macro F1)**---



---## Pipeline Overview



## Pipeline Overview| Phase | Notebook | Description |

|-------|----------|-------------|

| Phase | Notebook | Description | Best OOF F1 || 1 -- EDA & Baseline | [`eda_and_baseline.ipynb`](notebooks/eda_and_baseline.ipynb) | Class distribution, text statistics, TF-IDF + LinearSVC / LR baselines |

|-------|----------|-------------|-------------|| 2 -- Feature Engineering | [`feature_engineering.ipynb`](notebooks/feature_engineering.ipynb) | 46 stylometric features + TF-IDF (100 k dims), 10 classical models, 5-fold CV |

| 1 | [`eda_and_baseline.ipynb`](notebooks/eda_and_baseline.ipynb) | EDA, text statistics, TF-IDF + LR/SVC baselines | 0.9305 || 3 -- Transformer | [`transformer_finetuning.ipynb`](notebooks/transformer_finetuning.ipynb) | DeBERTa-v3-base with Focal Loss, LLRD, cosine scheduler, temperature scaling |

| 2 | [`feature_engineering.ipynb`](notebooks/feature_engineering.ipynb) | 46 stylometric features, 100k TF-IDF dims, 10 classical models | 0.9194 || 4 -- Ensemble Optimization | [`ensemble_optimization.ipynb`](notebooks/ensemble_optimization.ipynb) | Weighted soft voting (DeBERTa + SVC + LR), threshold optimization, pseudo-labeling |

| 3 | [`transformer_finetuning.ipynb`](notebooks/transformer_finetuning.ipynb) | DeBERTa-v3-base with Focal Loss, LLRD, temperature scaling | 0.9151 || 5 -- Advanced Ensemble | [`advanced_ensemble.ipynb`](notebooks/advanced_ensemble.ipynb) | Stacking meta-learner, rank averaging, per-class specialist voting, confidence-aware blending |

| 4 | [`ensemble_optimization.ipynb`](notebooks/ensemble_optimization.ipynb) | Weighted soft voting, threshold optimization, pseudo-labeling | 0.9552 || 6 -- Final Submission | [`final_submission.ipynb`](notebooks/final_submission.ipynb) | Score dashboard, submission agreement analysis, top-2 selection, reproducibility snapshot |

| 5 | [`advanced_ensemble.ipynb`](notebooks/advanced_ensemble.ipynb) | Stacking meta-learner, rank averaging, confidence-aware blending | 0.9552 |

| 6 | [`final_submission.ipynb`](notebooks/final_submission.ipynb) | Score dashboard, submission selection, reproducibility snapshot | -- |> Run notebooks in order: **Phase 1 > 2 > 3 > 4 > 5 > 6**. Each phase saves artifacts into `models/` that the next phase consumes.

| 7 | [`deberta_kfold_training.ipynb`](notebooks/deberta_kfold_training.ipynb) | DeBERTa 5-fold CV, weighted voting with SVC + LR | 0.9500 |

| 8 | [`ensemble_boosting.ipynb`](notebooks/ensemble_boosting.ipynb) | XGBoost + LightGBM diversity, 5-model ensemble, threshold tuning | 0.9591 |## Key Techniques



> **Execution order:** Phase 1 through 6 run sequentially (each saves artifacts consumed| Technique | Purpose |

> by the next). Phases 7 and 8 build on top of Phase 2 and 6 artifacts.|-----------|---------|

| Pre-tokenization | Tokenize once, reuse every epoch -- avoids per-batch tokenization bottleneck |

---| Layer-wise Learning Rate Decay (LLRD) | Lower LR for early transformer layers -- proven for DeBERTa fine-tuning |

| Focal Loss (γ = 2.0) + class weights | Handle severe class imbalance (19 : 1) |

## Key Techniques| Temperature scaling | Calibrate DeBERTa probabilities on a held-out calibration set |

| 4-way data split | Train / Val / Calibration / Test -- prevents information leakage |

| Technique | Purpose || Ensemble weight search | Grid-search optimal soft-voting weights on the calibration set |

|-----------|---------|| Per-class threshold optimization | Greedy multipliers that boost minority class recall |

| TF-IDF (word + char n-grams) | 100k sparse features capturing lexical and sub-word patterns || Stacking meta-learner | LR trained on 3 × 6 = 18 meta-features learns per-class model trust |

| 46 handcrafted features | Stylometric, readability, entropy, and AI-phrasing indicators || Pseudo-labeling | High-confidence test predictions augment the training set |

| DeBERTa-v3-base | Transformer fine-tuned with Focal Loss (gamma=2.0) and LLRD |

| 5-fold Stratified CV | Robust out-of-fold probability estimates for all models |## Project Structure

| Temperature scaling | Post-hoc calibration of DeBERTa logits |

| Weighted soft voting | Grid-search optimal blend of DeBERTa + SVC + LR + XGBoost + LightGBM |```

| Stacking meta-learner | LR/XGB trained on concatenated model probabilities |MALTO/

| Per-class threshold optimization | Greedy multipliers to boost minority-class recall |├── notebooks/

| Pseudo-labeling | High-confidence test predictions augment training data |│   ├── eda_and_baseline.ipynb           # Phase 1: EDA + TF-IDF baselines

│   ├── feature_engineering.ipynb        # Phase 2: 46 features + classical ML

---│   ├── transformer_finetuning.ipynb     # Phase 3: DeBERTa-v3-base fine-tuning

│   ├── ensemble_optimization.ipynb      # Phase 4: Basic ensemble + pseudo-labeling

## Project Structure│   ├── advanced_ensemble.ipynb          # Phase 5: Stacking + advanced ensemble

│   └── final_submission.ipynb           # Phase 6: Final polish + submission

```├── src/

MALTO/│   ├── __init__.py                      # Public API re-exports

├── notebooks/│   ├── features.py                      # extract_features() -- 46 features

│   ├── eda_and_baseline.ipynb           # Phase 1: EDA + TF-IDF baselines│   ├── models.py                        # TemperatureScaler, FocalLoss, ensemble utilities

│   ├── feature_engineering.ipynb        # Phase 2: 46 features + classical ML│   └── utils.py                         # Constants, data loading, submission helpers

│   ├── transformer_finetuning.ipynb     # Phase 3: DeBERTa fine-tuning├── figures/                             # All PNG visualizations (git-ignored, regenerated)

│   ├── ensemble_optimization.ipynb      # Phase 4: Ensemble + pseudo-labeling├── models/                              # Saved weights & artifacts (git-ignored)

│   ├── advanced_ensemble.ipynb          # Phase 5: Stacking + advanced ensemble├── checkpoints/                         # Model checkpoints for debugging (git-ignored)

│   ├── final_submission.ipynb           # Phase 6: Final submission pipeline├── submissions/                         # Kaggle submission CSVs (git-ignored)

│   ├── deberta_kfold_training.ipynb     # Phase 7: DeBERTa 5-fold CV├── train.csv / test.csv                 # Competition data (git-ignored)

│   └── ensemble_boosting.ipynb          # Phase 8: XGB/LGB ensemble diversity├── PROJECT_PLAN.md                      # Full 6-phase sprint plan

├── src/└── README.md                            # This file

│   ├── __init__.py```

│   ├── features.py                      # extract_features() -- 46 features

│   ├── models.py                        # FocalLoss, TemperatureScaler, ensemble utils## Dataset

│   └── utils.py                         # Constants, data I/O, submission helpers

├── models/                              # Saved weights and configs (git-ignored)| Class | Label | Train Samples | Share |

├── checkpoints/                         # Model checkpoints (git-ignored)|-------|-------|---------------|-------|

├── submissions/                         # Kaggle submission CSVs| Human | 0 | 1 520 | 63.3 % |

├── figures/                             # Visualization PNGs| ChatGPT | 5 | 320 | 13.3 % |

├── competition_info.json                # Competition metadata| Gemini | 4 | 240 | 10.0 % |

├── PROJECT_PLAN.md                      # Detailed phase-by-phase plan| Grok | 2 | 160 | 6.7 % |

└── README.md| DeepSeek | 1 | 80 | 3.3 % |

```| Claude | 3 | 80 | 3.3 % |



---## Environment



## Dataset- **Python 3.9** · PyTorch 2.8 · Transformers 4.57

- Memory-safe training settings: `BATCH_SIZE=2`, `GRAD_ACCUM=8` (effective batch = 16), `MAX_LEN=512`

| Class | Label | Train Samples | Share |

|-------|-------|---------------|-------|## Competition Rules

| Human | 0 | 1,520 | 63.3% |

| ChatGPT | 5 | 320 | 13.3% |- Pre-trained AI detectors forbidden

| Gemini | 4 | 240 | 10.0% |- External data forbidden

| Grok | 2 | 160 | 6.7% |- Individual competition only

| DeepSeek | 1 | 80 | 3.3% |
| Claude | 3 | 80 | 3.3% |

---

## Score Progression

| Submission | Method | Public LB |
|------------|--------|-----------|
| `01_tfidf_svc_baseline` | TF-IDF + LinearSVC | 0.84123 |
| `05_final_stacking` | Stacking meta-LR (DeBERTa + SVC + LR) | 0.90974 |
| `06_deberta_5fold` | DeBERTa 5-fold only | 0.91648 |
| `07_weighted_vote_best` | Weighted vote (DeBERTa 0.65 + SVC 0.25 + LR 0.10) | **0.92170** |

---

## Environment

- Python 3.9, PyTorch 2.8, Transformers 4.57
- scikit-learn 1.6, XGBoost 2.1, LightGBM 4.6
- Hardware: MacBook M4 Air (MPS backend)
- Training config: `BATCH_SIZE=2`, `GRAD_ACCUM=8` (effective 16), `MAX_LEN=512`

## Competition Rules

- Pre-trained AI detectors are forbidden
- External data is forbidden
- Individual competition only
