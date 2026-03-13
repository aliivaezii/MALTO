"""
Feature extraction utilities for AI text detection.

Extracts 46 linguistic, stylometric, and structural features from text
to distinguish human-written from AI-generated content.
"""

import re
import string
import numpy as np
import pandas as pd
from collections import Counter


def extract_features(texts, show_progress=True):
    """Extract 46 linguistic/stylometric features from a list of texts.
    
    Features include:
    - Basic length statistics (7)
    - Vocabulary richness (4) 
    - Character-level ratios (4)
    - Punctuation patterns (10)
    - Spelling/typo indicators (6)
    - Structural features (6)
    - AI-style indicators (4)
    - Readability scores (3)
    - Entropy measures (2)
    
    Parameters
    ----------
    texts : array-like of str
        Input texts to extract features from.
    show_progress : bool
        Whether to show a tqdm progress bar.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 46 feature columns, one row per text.
    """
    if show_progress:
        try:
            from tqdm.auto import tqdm
            texts = tqdm(texts, desc='Extracting features', leave=False)
        except ImportError:
            pass
    
    features = []
    for text in texts:
        f = {}
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # --- Basic length features (7) ---
        f['char_count'] = len(text)
        f['word_count'] = len(words)
        f['sentence_count'] = max(len(sentences), 1)
        f['avg_word_len'] = np.mean([len(w) for w in words]) if words else 0
        f['avg_sentence_len'] = f['word_count'] / f['sentence_count']
        f['max_word_len'] = max([len(w) for w in words]) if words else 0
        f['std_word_len'] = np.std([len(w) for w in words]) if len(words) > 1 else 0

        # --- Vocabulary richness (4) ---
        lower_words = [w.lower() for w in words]
        unique_words = set(lower_words)
        f['unique_word_count'] = len(unique_words)
        f['type_token_ratio'] = len(unique_words) / max(len(words), 1)
        word_freq = Counter(lower_words)
        f['hapax_ratio'] = sum(1 for v in word_freq.values() if v == 1) / max(len(unique_words), 1)
        freq_spectrum = Counter(word_freq.values())
        N = len(lower_words)
        M2 = sum(i * i * freq_spectrum[i] for i in freq_spectrum)
        f['yules_k'] = 10000 * (M2 - N) / (N * N) if N > 1 else 0

        # --- Character-level features (4) ---
        f['upper_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        f['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        f['space_ratio'] = sum(1 for c in text if c.isspace()) / max(len(text), 1)
        f['alpha_ratio'] = sum(1 for c in text if c.isalpha()) / max(len(text), 1)

        # --- Punctuation features (10) ---
        f['punct_ratio'] = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
        f['comma_ratio'] = text.count(',') / max(len(words), 1)
        f['period_ratio'] = text.count('.') / max(len(words), 1)
        f['exclamation_ratio'] = text.count('!') / max(len(words), 1)
        f['question_ratio'] = text.count('?') / max(len(words), 1)
        f['semicolon_ratio'] = text.count(';') / max(len(words), 1)
        f['colon_ratio'] = text.count(':') / max(len(words), 1)
        f['quote_count'] = text.count('"') + text.count("'")
        f['paren_count'] = text.count('(') + text.count(')')
        f['dash_count'] = text.count('-') + text.count('\u2014')

        # --- Spelling / typo indicators (6) ---
        f['double_letter_ratio'] = len(re.findall(r'(.)\1', text.lower())) / max(len(words), 1)
        f['short_word_ratio'] = sum(1 for w in words if len(w) <= 2) / max(len(words), 1)
        f['long_word_ratio'] = sum(1 for w in words if len(w) > 10) / max(len(words), 1)
        f['very_long_word_ratio'] = sum(1 for w in words if len(w) > 15) / max(len(words), 1)
        consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', text.lower())
        f['consonant_cluster_ratio'] = len(consonant_clusters) / max(len(words), 1)
        repeated_chars = re.findall(r'(.)\1{2,}', text)
        f['repeated_char_count'] = len(repeated_chars)

        # --- Structural features (6) ---
        f['paragraph_count'] = text.count('\n\n') + 1
        f['newline_count'] = text.count('\n')
        f['starts_with_upper'] = int(text[0].isupper()) if text else 0
        f['ends_with_period'] = int(text.rstrip().endswith('.')) if text else 0
        f['bullet_count'] = len(re.findall(r'^\s*[-*\u2022]\s', text, re.MULTILINE))
        f['numbered_list_count'] = len(re.findall(r'^\s*\d+[.):]\s', text, re.MULTILINE))

        # --- AI-style indicators (4) ---
        text_lower = text.lower()
        ai_phrases = [
            'in conclusion', 'furthermore', 'moreover', 'additionally',
            'it is worth noting', 'it is important to', 'on the other hand',
            'in summary', 'to summarize', 'in essence', 'ultimately',
            'delve', 'crucial', 'comprehensive', 'leverage', 'facilitate',
            'robust', 'streamline', 'innovative', 'cutting-edge', 'paradigm',
            'holistic', 'synergy', 'encompass', 'multifaceted', 'nuanced'
        ]
        f['ai_phrase_count'] = sum(1 for p in ai_phrases if p in text_lower)

        transitions = [
            'however', 'therefore', 'nonetheless', 'nevertheless',
            'consequently', 'meanwhile', 'subsequently', 'accordingly',
            'hence', 'thus', 'thereby'
        ]
        f['transition_count'] = sum(1 for t in transitions if t in text_lower)

        first_person = re.findall(
            r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b', text_lower
        )
        f['first_person_ratio'] = len(first_person) / max(len(words), 1)

        contractions = re.findall(r"\b\w+'\w+\b", text)
        f['contraction_ratio'] = len(contractions) / max(len(words), 1)

        # --- Readability proxies (3) ---
        def _count_syllables(word):
            word = word.lower().rstrip('e')
            return max(1, len(re.findall(r'[aeiouy]+', word)))

        total_syllables = sum(_count_syllables(w) for w in words) if words else 1
        f['flesch_reading_ease'] = (
            206.835
            - 1.015 * f['avg_sentence_len']
            - 84.6 * (total_syllables / max(len(words), 1))
        )
        f['flesch_kincaid_grade'] = (
            0.39 * f['avg_sentence_len']
            + 11.8 * (total_syllables / max(len(words), 1))
            - 15.59
        )
        f['ari'] = (
            4.71 * (f['char_count'] / max(len(words), 1))
            + 0.5 * f['avg_sentence_len']
            - 21.43
        )

        # --- Entropy features (2) ---
        char_freq = Counter(text.lower())
        total_chars = sum(char_freq.values())
        char_probs = [c / total_chars for c in char_freq.values()]
        f['char_entropy'] = -sum(p * np.log2(p) for p in char_probs if p > 0)

        word_probs = [c / len(lower_words) for c in word_freq.values()] if lower_words else [1]
        f['word_entropy'] = -sum(p * np.log2(p) for p in word_probs if p > 0)

        features.append(f)

    return pd.DataFrame(features)


FEATURE_NAMES = [
    # Basic length (7)
    'char_count', 'word_count', 'sentence_count', 'avg_word_len',
    'avg_sentence_len', 'max_word_len', 'std_word_len',
    # Vocabulary richness (4)
    'unique_word_count', 'type_token_ratio', 'hapax_ratio', 'yules_k',
    # Character-level (4)
    'upper_ratio', 'digit_ratio', 'space_ratio', 'alpha_ratio',
    # Punctuation (10)
    'punct_ratio', 'comma_ratio', 'period_ratio', 'exclamation_ratio',
    'question_ratio', 'semicolon_ratio', 'colon_ratio', 'quote_count',
    'paren_count', 'dash_count',
    # Spelling/typo (6)
    'double_letter_ratio', 'short_word_ratio', 'long_word_ratio',
    'very_long_word_ratio', 'consonant_cluster_ratio', 'repeated_char_count',
    # Structural (6)
    'paragraph_count', 'newline_count', 'starts_with_upper',
    'ends_with_period', 'bullet_count', 'numbered_list_count',
    # AI-style (4)
    'ai_phrase_count', 'transition_count', 'first_person_ratio',
    'contraction_ratio',
    # Readability (3)
    'flesch_reading_ease', 'flesch_kincaid_grade', 'ari',
    # Entropy (2)
    'char_entropy', 'word_entropy',
]
