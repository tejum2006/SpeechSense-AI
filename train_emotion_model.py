"""
SpeechSense AI - Emotion Model Training Pipeline
================================================
Trains both:
1. Histogram Gradient Boosting (Modern SOTA tree ensemble)
2. Random Forest Classifier (Academic baseline)
on the RAVDESS dataset using strict actor-independent splitting to prevent identity leakage.

Artifacts Generated:
- models/emotion_model.pkl (Bundled Scaler + Both Classifiers)
- models/evaluation_report.json (Genuine reproducible metrics for both models)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Configure environment before heavy imports to avoid sandbox locks
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_NUMBA_DIR = _CACHE_DIR / "numba"
_MPL_DIR = _CACHE_DIR / "matplotlib"
_NUMBA_DIR.mkdir(parents=True, exist_ok=True)
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from feature_extraction import (
    scan_ravdess_dataset,
    extract_features_from_audio,
    get_feature_names,
    EMOTION_LABELS,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "dataset" / "RAVDESS"
DEFAULT_MODELS_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = DEFAULT_MODELS_DIR / "emotion_model.pkl"
REPORT_SAVE_PATH = DEFAULT_MODELS_DIR / "evaluation_report.json"

DEFAULT_TEST_ACTORS = [21, 22, 23, 24]


def extract_features_from_dataset(
    sample_records: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    actors_list: List[int] = []

    total = len(sample_records)
    print(f"\n[+] Extracting acoustic features for {total} speech audio files...")

    for idx, rec in enumerate(sample_records, 1):
        if idx % 50 == 0 or idx == total:
            print(f"    Progress: {idx}/{total} ({(idx/total)*100:.1f}%)")

        try:
            feats = extract_features_from_audio(rec["file_path"])
            X_list.append(feats)
            y_list.append(rec["emotion"])
            actors_list.append(rec["actor_id"])
        except Exception as err:
            print(f"    [!] Warning: Failed processing '{rec['filename']}': {err}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=object)
    actor_ids = np.array(actors_list, dtype=int)

    return X, y, actor_ids


def train_and_evaluate(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    test_actors: List[int] = DEFAULT_TEST_ACTORS,
    random_state: int = 42,
) -> Dict[str, Any]:
    print("=" * 70)
    print("   SPEECHSENSE AI - DUAL SER MODEL TRAINING PIPELINE")
    print("=" * 70)

    samples = scan_ravdess_dataset(dataset_dir, speech_only=True)
    if not samples:
        print(f"ERROR: No valid RAVDESS speech audio files found in {dataset_dir}")
        return {"status": "error"}

    print(f"[+] Found {len(samples)} valid speech audio files across all actors.")
    X, y, actor_ids = extract_features_from_dataset(samples)

    # Actor-independent split
    test_mask = np.isin(actor_ids, test_actors)
    train_mask = ~test_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Verify zero leakage
    overlap = set(actor_ids[train_mask]).intersection(set(actor_ids[test_mask]))
    if overlap:
        raise AssertionError(f"Actor leakage detected: {overlap}")
    print(f"[✓] Zero actor overlap: {len(X_train)} train samples, {len(X_test)} test samples.")

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Train Random Forest (Academic Baseline)
    print("\n[*] Training Model 1: RandomForestClassifier (150 trees)...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=4,
    )
    rf.fit(X_train_scaled, y_train)

    # 2. Train HistGradientBoosting (Modern SOTA Ensemble)
    print("[*] Training Model 2: HistGradientBoostingClassifier (Modern Gradient Boosting)...")
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=8,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=random_state,
    )
    hgb.fit(X_train_scaled, y_train)

    # Evaluate both models
    results = {}
    labels_order = [l for l in EMOTION_LABELS if l in rf.classes_]

    for name, model in [("Gradient Boosting (Latest)", hgb), ("Random Forest (Baseline)", rf)]:
        preds = model.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, preds))
        f1_m = float(f1_score(y_test, preds, average="macro", zero_division=0))
        prec_m = float(precision_score(y_test, preds, average="macro", zero_division=0))
        rec_m = float(recall_score(y_test, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, preds, labels=labels_order).tolist()

        results[name] = {
            "accuracy": acc,
            "precision_macro": prec_m,
            "recall_macro": rec_m,
            "f1_macro": f1_m,
            "confusion_matrix": cm,
        }
        print(f"\n--- {name} Results ---")
        print(f"Accuracy: {acc*100:.2f}% | Macro F1: {f1_m*100:.2f}%")

    # Serialize both models into bundle
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "scaler": scaler,
        "classifier": hgb,  # Default to latest gradient boosting
        "classifiers": {
            "Gradient Boosting (Latest)": hgb,
            "Random Forest (Baseline)": rf,
        },
        "classes": labels_order,
        "feature_names": get_feature_names(),
        "test_actors": test_actors,
    }
    joblib.dump(bundle, MODEL_SAVE_PATH)
    print(f"\n[+] Saved dual-model artifact to: {MODEL_SAVE_PATH.resolve()}")

    # Save full evaluation report
    report = {
        "dataset": "RAVDESS (Speech)",
        "test_actors": test_actors,
        "train_samples_count": int(len(X_train)),
        "test_samples_count": int(len(X_test)),
        "classes": labels_order,
        "models": results,
        "primary_model": "Gradient Boosting (Latest)",
        "accuracy": results["Gradient Boosting (Latest)"]["accuracy"],
        "precision_macro": results["Gradient Boosting (Latest)"]["precision_macro"],
        "recall_macro": results["Gradient Boosting (Latest)"]["recall_macro"],
        "f1_macro": results["Gradient Boosting (Latest)"]["f1_macro"],
        "confusion_matrix": results["Gradient Boosting (Latest)"]["confusion_matrix"],
    }
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved evaluation report to: {REPORT_SAVE_PATH.resolve()}\n")

    return report


if __name__ == "__main__":
    train_and_evaluate()
