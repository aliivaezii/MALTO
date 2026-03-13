# MALTO Recruitment Hackathon -- Project Plan

## Human vs. AI-Generated Text Classification

---

## Competition Summary

| Item | Detail |
|------|--------|
| **Task** | 6-class text classification (Human + 5 AI models) |
| **Metric** | Macro F1 Score |
| **Train** | 2,400 samples (heavily imbalanced) |
| **Test** | 600 samples |
| **Submission** | `ID,LABEL` -- 600 rows, integer labels 0-5 |
| **Max Subs** | 10/day, 2 final selections |
| **Forbidden** | Pre-trained AI detectors, external data |

### Class Distribution

```
Label 0 (Human):    1520 samples  (63.3%)
Label 5 (ChatGPT):   320 samples  (13.3%)
Label 4 (Gemini):    240 samples  (10.0%)
Label 2 (Grok):      160 samples   (6.7%)
Label 1 (DeepSeek):   80 samples   (3.3%)
Label 3 (Claude):     80 samples   (3.3%)
```

---

## 8-Phase Pipeline

### Phase 1 -- EDA + Baseline

**Notebook:** `notebooks/eda_and_baseline.ipynb`

- [x] Text length, vocabulary, n-gram, misspelling analysis
- [x] TF-IDF (word + char) + Logistic Regression + LinearSVC
- [x] First Kaggle submission

| Result | Score |
|--------|-------|
| TF-IDF + LR (val split) | **0.9305** |
| TF-IDF + SVC (5-fold CV) | 0.8935 |

---

### Phase 2 -- Feature Engineering + Classical ML

**Notebook:** `notebooks/feature_engineering.ipynb`

- [x] 46 handcrafted linguistic/stylometric features
- [x] TF-IDF: word (1-2gram, 50k) + char_wb (2-5gram, 50k) = 100k features
- [x] 10 classifiers: LR x4, SVC x4, XGBoost, LightGBM
- [x] 5-fold Stratified CV on all models

| Result | Score |
|--------|-------|
| **SVC (C=5.0)** | **0.9194** (CV) |
| SVC (C=2.0) | 0.9192 |
| LightGBM | 0.9150 |

---

### Phase 3 -- Transformer Fine-Tuning

**Notebook:** `notebooks/transformer_finetuning.ipynb`

- [x] DeBERTa-v3-base + Focal Loss (gamma=2.0) + class weights
- [x] LLRD (factor=0.9) + cosine scheduler + 10% warmup
- [x] BATCH=2 x GRAD_ACCUM=8 (effective=16), MAX_LEN=512
- [x] Train/Val/Cal split (70/15/15) + early stopping (patience=2)
- [x] Temperature scaling on calibration set

---

### Phase 4 -- Ensemble Optimization

**Notebook:** `notebooks/ensemble_optimization.ipynb`

- [x] Re-train SVC + LR on same split as DeBERTa
- [x] Error analysis: DeBERTa vs SVC vs LR disagreement
- [x] Weighted soft voting (grid search on calibration set)
- [x] Per-class threshold optimization
- [x] Pseudo-labeling (high-confidence test samples)
- [x] 5-fold CV for SVC + LR (robust OOF probs)

---

### Phase 5 -- Advanced Ensemble

**Notebook:** `notebooks/advanced_ensemble.ipynb`

- [x] Stacking meta-learner (LR on 3x6=18 meta-features)
- [x] Rank averaging (robust to miscalibration)
- [x] Per-class specialist voting
- [x] Confidence-aware blending (dynamic per-sample weights)
- [x] Threshold-optimized stacking

---

### Phase 6 -- Final Submission

**Notebook:** `notebooks/final_submission.ipynb`

- [x] Inventory all submission files
- [x] Score progression dashboard
- [x] Submission agreement analysis (pairwise heatmap)
- [x] Select top-2 final submissions
- [x] Reproducibility snapshot

---

### Phase 7 -- DeBERTa 5-Fold CV

**Notebook:** `notebooks/deberta_kfold_training.ipynb`

- [x] 5-fold stratified DeBERTa-v3-base training (Focal Loss + LLRD)
- [x] Temperature scaling per fold
- [x] Weighted voting: DeBERTa 0.65 + SVC 0.25 + LR 0.10
- [x] Best OOF F1: 0.9500, LB: **0.92170**

---

### Phase 8 -- Ensemble Boosting

**Notebook:** `notebooks/ensemble_boosting.ipynb`

- [x] XGBoost 5-fold OOF (F1=0.9180)
- [x] LightGBM 5-fold OOF (F1=0.9165)
- [x] 5-model weight search (DeBERTa + SVC + LR + XGB + LGB)
- [x] Stacking with multiple meta-learners (LR, XGB, LGB)
- [x] Per-class threshold optimization
- [x] Best OOF F1: 0.9591

---

## Key Technical Strategies

1. **Misspelling features** -- Human texts have typos; AI texts are clean
2. **Class weighting** -- `class_weight='balanced'` everywhere (Macro F1 on 19:1 imbalance)
3. **Stratified CV** -- Never use random splits with this level of imbalance
4. **TF-IDF + SVM** -- Surprisingly strong for text classification
5. **DeBERTa + Focal Loss + LLRD** -- State-of-the-art transformer approach
6. **Stacking** -- Meta-learner that learns per-class model trust
7. **Threshold optimization** -- Per-class multipliers to boost minority recall
8. **Pseudo-labeling** -- Semi-supervised learning on high-confidence test predictions

---

## Execution Order

Run notebooks in this order (each depends on artifacts from the previous):

```
Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8
```

Phases 1 and 2 are independent of each other (both read `train.csv` / `test.csv` directly).
Phase 3 onward requires artifacts from earlier phases saved in `models/`.
