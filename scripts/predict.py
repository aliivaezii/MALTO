"""
CLI inference script — MALTO AI text source detection.

Uses the saved TF-IDF vectorizers + Calibrated LinearSVC from malto_model/.
Does NOT require the transformer fold checkpoints (SVC-only prediction).

Usage
-----
    python scripts/predict.py --text "Your text here"
    python scripts/predict.py --file path/to/texts.txt
    python scripts/predict.py --text "..." --top3

Example
-------
    $ python scripts/predict.py --text "The mitochondria is the powerhouse of the cell."
    Predicted class : Human
    Confidence      : 0.934
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR  = REPO_ROOT / "malto_model"
SRC_DIR    = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))

from src.features import extract_features  # noqa: E402


def load_model():
    config   = json.loads((MODEL_DIR / "ensemble_config.json").read_text())
    char_vec = joblib.load(MODEL_DIR / "char_tfidf.pkl")
    word_vec = joblib.load(MODEL_DIR / "word_tfidf.pkl")
    svc      = joblib.load(MODEL_DIR / "svc_model.pkl")
    labels   = [config["label_map"][str(i)] for i in range(config["num_labels"])]
    return char_vec, word_vec, svc, labels


def predict(texts, char_vec, word_vec, svc, labels, top3=False):
    X = hstack([char_vec.transform(texts), word_vec.transform(texts)])
    probs = svc.predict_proba(X)
    results = []
    for prob in probs:
        top_idx = int(np.argmax(prob))
        entry = {
            "prediction": labels[top_idx],
            "confidence": round(float(prob[top_idx]), 4),
        }
        if top3:
            ranked = sorted(enumerate(prob), key=lambda x: -x[1])[:3]
            entry["top3"] = [{"class": labels[i], "prob": round(float(p), 4)} for i, p in ranked]
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict AI text source (SVC model).")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text string to classify")
    group.add_argument("--file", type=str, help="Path to a plain-text file (one text per line)")
    parser.add_argument("--top3", action="store_true", help="Show top-3 class probabilities")
    args = parser.parse_args()

    texts = [args.text] if args.text else Path(args.file).read_text().splitlines()
    texts = [t for t in texts if t.strip()]

    char_vec, word_vec, svc, labels = load_model()
    results = predict(texts, char_vec, word_vec, svc, labels, top3=args.top3)

    for text, res in zip(texts, results):
        if len(texts) > 1:
            print(f"\nText    : {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Predicted class : {res['prediction']}")
        print(f"Confidence      : {res['confidence']:.3f}")
        if args.top3:
            for rank, item in enumerate(res["top3"], 1):
                print(f"  #{rank} {item['class']:<10} {item['prob']:.4f}")


if __name__ == "__main__":
    main()
