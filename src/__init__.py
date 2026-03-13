"""
MALTO — Reusable utilities for the Human vs. AI Text Classification pipeline.

Modules
-------
features : Feature extraction (46 stylometric features)
models   : Temperature scaling, Focal Loss, ensemble search, threshold optimization
utils    : Constants, data I/O, submission helpers, artifact I/O
"""

from .features import extract_features, FEATURE_NAMES
from .models import TemperatureScaler, FocalLoss, search_ensemble_weights, optimize_thresholds
from .utils import (
    SEED, NUM_LABELS, LABEL_MAP, LABEL_MAP_INV,
    load_data, save_submission, save_submission_variants,
    save_artifacts, load_artifacts, save_config, load_config,
)
