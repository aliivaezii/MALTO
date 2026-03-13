"""
Model training utilities for the MALTO competition.

Includes calibration, ensemble weight search, threshold optimization,
and temperature scaling.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Post-hoc temperature scaling for probability calibration.
    
    Fits a single scalar temperature T on a held-out calibration set
    so that softmax(logits / T) produces well-calibrated probabilities.
    """
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels, t_range=(0.5, 5.0), step=0.05, verbose=True):
        """Find optimal temperature on calibration set via grid search.
        
        Parameters
        ----------
        logits : np.ndarray, shape (n_samples, n_classes)
            Raw model logits.
        labels : np.ndarray, shape (n_samples,)
            True labels.
        t_range : tuple
            (min_temp, max_temp) to search.
        step : float
            Grid step size.
        verbose : bool
            Whether to print results.
        
        Returns
        -------
        self
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        
        best_temp, best_nll = 1.0, float('inf')
        for temp in np.arange(t_range[0], t_range[1], step):
            nll = nn.functional.cross_entropy(logits_t / temp, labels_t).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
        
        if verbose:
            preds_before = logits_t.argmax(-1).numpy()
            preds_after = (logits_t / self.temperature).argmax(-1).numpy()
            f1_before = f1_score(labels, preds_before, average='macro')
            f1_after = f1_score(labels, preds_after, average='macro')
            print(f'  Temperature: {self.temperature:.2f}')
            print(f'  Cal NLL:     {best_nll:.4f}')
            print(f'  Cal F1:      {f1_before:.4f} → {f1_after:.4f}')
        
        return self
    
    def predict_proba(self, logits):
        """Get calibrated probabilities.
        
        Parameters
        ----------
        logits : np.ndarray, shape (n_samples, n_classes)
        
        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Softmax probabilities after temperature scaling.
        """
        scaled = torch.tensor(logits, dtype=torch.float32) / self.temperature
        return torch.softmax(scaled, dim=-1).numpy()


# ---------------------------------------------------------------------------
# Ensemble Weight Optimization
# ---------------------------------------------------------------------------

def search_ensemble_weights(prob_list, labels, step=0.05):
    """Grid-search optimal weights for soft-voting ensemble.
    
    Searches over all weight triplets (for 3 models) that sum to 1.0,
    using the given labels to maximize macro F1.
    
    Parameters
    ----------
    prob_list : list of np.ndarray
        List of probability arrays, each shape (n_samples, n_classes).
        Currently supports 2 or 3 models.
    labels : np.ndarray
        True labels.
    step : float
        Grid step size.
    
    Returns
    -------
    best_weights : tuple of float
        Optimal weights.
    best_f1 : float
        Best macro F1 achieved.
    """
    n_models = len(prob_list)
    best_f1 = 0
    best_weights = tuple(1.0 / n_models for _ in range(n_models))
    
    if n_models == 2:
        for w0 in np.arange(0.0, 1.01, step):
            w1 = 1.0 - w0
            ens = w0 * prob_list[0] + w1 * prob_list[1]
            f1 = f1_score(labels, ens.argmax(axis=1), average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w0, w1)
    
    elif n_models == 3:
        for w0 in np.arange(0.0, 1.01, step):
            for w1 in np.arange(0.0, 1.01 - w0, step):
                w2 = 1.0 - w0 - w1
                if w2 < -0.01:
                    continue
                w2 = max(w2, 0.0)
                ens = w0 * prob_list[0] + w1 * prob_list[1] + w2 * prob_list[2]
                f1 = f1_score(labels, ens.argmax(axis=1), average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = (w0, w1, w2)
    else:
        raise ValueError(f'Supports 2 or 3 models, got {n_models}')
    
    return best_weights, best_f1


# ---------------------------------------------------------------------------
# Per-Class Threshold Optimization
# ---------------------------------------------------------------------------

def optimize_thresholds(probs, labels, n_classes=6, steps=50, n_passes=3):
    """Find per-class multipliers that maximize macro F1.
    
    Instead of argmax(probs), computes argmax(probs * multipliers)
    where multipliers[c] > 1 boosts class c.
    
    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_classes)
    labels : np.ndarray, shape (n_samples,)
    n_classes : int
    steps : int
        Number of grid points for each multiplier.
    n_passes : int
        Number of greedy optimization passes.
    
    Returns
    -------
    multipliers : np.ndarray, shape (n_classes,)
    best_f1 : float
    """
    best_multipliers = np.ones(n_classes)
    best_f1 = f1_score(labels, probs.argmax(axis=1), average='macro')
    
    for _ in range(n_passes):
        improved = False
        for cls in range(n_classes):
            for mult in np.linspace(0.5, 2.0, steps):
                trial = best_multipliers.copy()
                trial[cls] = mult
                preds = (probs * trial).argmax(axis=1)
                f1 = f1_score(labels, preds, average='macro')
                if f1 > best_f1 + 1e-6:
                    best_f1 = f1
                    best_multipliers = trial.copy()
                    improved = True
        if not improved:
            break
    
    return best_multipliers, best_f1


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.
    
    Reduces the loss contribution from easy examples,
    focusing training on hard/misclassified samples.
    
    Parameters
    ----------
    alpha : torch.Tensor or None
        Per-class weights.
    gamma : float
        Focusing parameter (0 = standard CE, 2 = typical).
    """
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()
