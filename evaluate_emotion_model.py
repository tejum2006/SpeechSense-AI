"""
SpeechSense AI - Emotion Model Evaluation Module
================================================
Inspects and visualizes saved evaluation metrics and confusion matrices.
Generates a publishable confusion matrix plot in assets/confusion_matrix.png.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_MPL_DIR = _CACHE_DIR / "matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "models" / "evaluation_report.json"
ASSETS_DIR = BASE_DIR / "assets"
CM_FIGURE_PATH = ASSETS_DIR / "confusion_matrix.png"


def display_and_export_evaluation(report_path: Path = REPORT_PATH) -> Optional[Dict[str, Any]]:
    """
    Reads evaluation_report.json, prints formatted performance metrics,
    and plots the 8x8 confusion matrix.
    """
    if not report_path.exists():
        print(f"[!] Evaluation report not found at: {report_path.resolve()}")
        print("    Please run: python train_emotion_model.py")
        return None

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 65)
    print("      SPEECHSENSE AI - SER PERFORMANCE EVALUATION REPORT")
    print("=" * 65)
    print(f"Model Architecture : {data.get('model_type', 'Random Forest')}")
    print(f"Dataset            : {data.get('dataset', 'RAVDESS')}")
    print(f"Test Actors        : {data.get('test_actors', [])}")
    print(f"Train / Test Split : {data.get('train_samples_count')} train / {data.get('test_samples_count')} test")
    print("-" * 65)
    print(f"Overall Accuracy   : {data.get('accuracy', 0.0) * 100:.2f}%")
    print(f"Macro Precision    : {data.get('precision_macro', 0.0) * 100:.2f}%")
    print(f"Macro Recall       : {data.get('recall_macro', 0.0) * 100:.2f}%")
    print(f"Macro F1-Score     : {data.get('f1_macro', 0.0) * 100:.2f}%")
    print(f"Weighted F1-Score  : {data.get('f1_weighted', 0.0) * 100:.2f}%")
    print("=" * 65)

    # Render confusion matrix if available
    cm = np.array(data.get("confusion_matrix", []))
    classes = data.get("classes", [])

    if len(cm) > 0 and len(classes) > 0:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 7))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("SpeechSense AI - 8-Class Emotion Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha="right")
        plt.yticks(tick_marks, classes)

        # Annotate cells
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                plt.text(
                    j,
                    i,
                    format(val, "d"),
                    horizontalalignment="center",
                    color="white" if val > thresh else "#173B68",
                    fontweight="bold" if val > 0 else "normal",
                )

        plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
        plt.tight_layout()

        plt.savefig(CM_FIGURE_PATH, dpi=300)
        plt.close()
        print(f"[+] Saved confusion matrix visualization to: {CM_FIGURE_PATH.resolve()}")

    return data


if __name__ == "__main__":
    display_and_export_evaluation()
