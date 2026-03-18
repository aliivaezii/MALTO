# Technical Write-up — MALTO Hackathon (2nd Place)

## Problem

The task is to classify a piece of text as either human-written or identify which AI model produced it — one of six classes: **Human, DeepSeek, Grok, Claude, Gemini, ChatGPT**.

The dataset has 2,400 training and 600 test samples. The core difficulty is severe class imbalance: Human accounts for 63% of the data, while DeepSeek and Claude each have only 80 samples (3.3%) — a 19:1 ratio. The evaluation metric is macro F1, so every class carries equal weight regardless of sample count. Getting the rare classes right matters as much as the common ones.

## Approach

### Two-model stack

The solution combines a fine-tuned transformer with a classical n-gram model, blended via an ensemble optimised on out-of-fold predictions.

**Transformer (ModernBERT-base):** ModernBERT was chosen over DeBERTa because it trains faster on Kaggle's T4×2 budget while achieving comparable accuracy for this input length (≤512 tokens). Training uses 5-fold cross-validation plus a separate full-data model blended at α=0.6 with the fold average — this gives a mild boost over either alone.

**Classical model (TF-IDF + LinearSVC):** Character 3–5 grams and word 1–2 grams (100k features total) fed into a calibrated LinearSVC (C=5). TF-IDF captures stylistic fingerprints that differ across AI providers — things like punctuation density, sentence-ending patterns, and specific phrase choices. The SVC is calibrated with `CalibratedClassifierCV` to produce reliable probabilities for blending.

### Handling class imbalance

Standard cross-entropy with inverse-frequency weights causes instability for very small classes. Instead:

- **LDAM loss** (Label-Distribution-Aware Margin) adds a per-class margin proportional to 1/n_c^0.25, so rare classes are pushed farther from the decision boundary.
- **Deferred Re-Weighting (DRW)** activates class-weighted loss from epoch 2 onward rather than from epoch 1. This gives the model a clean initialisation phase before tilting toward rare classes.
- **Weight cap at 20×**: Uncapping weights causes training instability; 20× lets DeepSeek receive ~19× its natural weight without destabilising the loss scale.

### Ensemble optimisation

The transformer and SVC predictions are blended on a per-class basis using Nelder-Mead optimisation on out-of-fold predictions. This is the most important design decision in the solution.

A naive global blend (single weight for all classes) gives SVC roughly equal influence on every class. But the SVC has very different reliability across classes — it achieves F1 ≈ 0.51 on DeepSeek, while the transformer reaches 0.84. A global blend of 70/30 halves DeepSeek performance. Per-class weights let DeepSeek use 100% transformer while Grok and Gemini draw moderate SVC signal.

12 random initialisations are used to reduce sensitivity to local optima in Nelder-Mead. The best result across all starts is kept.

### Post-processing

After ensemble blending, temperature scaling calibrates the softmax distribution (T=0.30 — quite low, indicating the raw logits are underconfident). A greedy per-class threshold sweep over [0.85, 1.20] on the OOF predictions further nudges borderline predictions.

## Key insights

**1. The ensemble strategy matters more than the model.** The same ModernBERT weights, blended globally, score 0.94074. With per-class Nelder-Mead, the score jumps to 0.95341 — a 1.3 point gap from a single algorithmic change.

**2. TF-IDF is unreliable for DeepSeek/Grok distinction.** Character n-grams cannot separate these two models reliably. The SVC predicts 50 Grok and 8 DeepSeek on the 600-sample test set, while the correct distribution is ≈40 and ≈20 respectively. All 20 SVC-vs-transformer disagreements were DeepSeek/Grok confusions. When these two models disagree, the transformer is right.

**3. Minority class performance is the ceiling, not the floor.** Human achieves F1=1.00 regardless of approach. The macro F1 ceiling is set entirely by DeepSeek and Grok. Every technique decision — loss function, ensemble, threshold sweep — was evaluated primarily by its effect on these two classes.

**4. Loss function hierarchy for long-tail data:** LDAM+DRW > uncapped FocalLoss > capped FocalLoss (5× cap). The 5× cap was the root cause of an early regression: DeepSeek has a natural imbalance of 19× but received only 0.5× effective weight under the cap.

## What didn't work

- **R-Drop regularisation**: Running two forward passes with different dropout masks and adding a KL term did not improve OOF F1 measurably, and added ~50% training time.
- **KNN-based label corrections**: TF-IDF character n-gram KNN was used to identify potential mislabelled test predictions. All six corrections made the public score worse, confirming that TF-IDF cannot override transformer predictions for author attribution.
- **DeBERTa-v3-base**: Competitive on OOF but slower to train; the marginal quality gain over ModernBERT did not justify the training budget on T4×2.
- **Stacking meta-learner**: A LightGBM meta-model trained on OOF predictions from 6 base models was slightly outperformed by the Nelder-Mead blend on the public leaderboard.

## Results

| Stage | Public LB (Macro F1) |
|---|---|
| TF-IDF + LinearSVC baseline | 0.84123 |
| DeBERTa 5-fold | 0.91648 |
| ModernBERT + LDAM + DRW (3-fold) | 0.94120 |
| ModernBERT + SVC ensemble (per-class Nelder-Mead) | 0.95341 |
| **Final submission** | **0.95919** |

**2nd place** out of all participating teams. 1st place scored 0.96423.
