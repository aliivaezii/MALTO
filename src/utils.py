"""
Data loading and submission utilities for the MALTO competition.
"""

import os
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
NUM_LABELS = 6
LABEL_MAP = {
    0: 'Human',
    1: 'DeepSeek',
    2: 'Grok',
    3: 'Claude',
    4: 'Gemini',
    5: 'ChatGPT',
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_data(data_dir='.'):
    """Load train and test DataFrames.
    
    Returns
    -------
    train_df : pd.DataFrame
        Training data with columns TEXT, LABEL.
    test_df : pd.DataFrame
        Test data with column TEXT.
    """
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return train_df, test_df


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

def save_submission(preds, path, header='ID,LABEL'):
    """Save predictions as a Kaggle submission CSV.
    
    Parameters
    ----------
    preds : array-like of int
        Predicted labels (length 600 for this competition).
    path : str
        Output file path.
    header : str
        Header line for the CSV.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lines = [header] + [f'{i},{int(preds[i])}' for i in range(len(preds))]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def save_submission_variants(preds, basename, submissions_dir='submissions'):
    """Save predictions as a single submission CSV with the required format.
    
    Parameters
    ----------
    preds : array-like of int
        Predicted labels.
    basename : str
        Base filename (without extension), e.g. 'deberta_calibrated'.
    submissions_dir : str
        Directory to save into.
    """
    os.makedirs(submissions_dir, exist_ok=True)
    fname = f'{basename}.csv'
    save_submission(preds, os.path.join(submissions_dir, fname), header='ID,LABEL')
    print(f'  {fname:40s} header: ID,LABEL')


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------

def save_artifacts(artifacts_dict, models_dir='models'):
    """Save a dict of {name: np.array} as .npy files.
    
    Parameters
    ----------
    artifacts_dict : dict
        Mapping of artifact name to numpy array.
    models_dir : str
        Directory to save into.
    """
    os.makedirs(models_dir, exist_ok=True)
    for name, arr in artifacts_dict.items():
        path = os.path.join(models_dir, f'{name}.npy')
        np.save(path, arr)
        print(f'  Saved {name}: {arr.shape}')


def load_artifacts(names, models_dir='models'):
    """Load a list of .npy artifacts by name.
    
    Parameters
    ----------
    names : list of str
        Artifact names (without .npy extension).
    models_dir : str
        Directory to load from.
    
    Returns
    -------
    dict
        Mapping of name to numpy array.
    """
    result = {}
    for name in names:
        path = os.path.join(models_dir, f'{name}.npy')
        result[name] = np.load(path)
    return result


def save_config(config, path='models/config.json'):
    """Save a configuration dict as JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  Saved config to {path}')


def load_config(path='models/config.json'):
    """Load a configuration dict from JSON."""
    with open(path, 'r') as f:
        return json.load(f)
