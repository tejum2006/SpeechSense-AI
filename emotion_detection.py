"""
SpeechSense AI - Emotion Detection Inference Module
===================================================
Loads the trained SER models (Histogram Gradient Boosting & Random Forest)
and feature scaler, extracts 61 acoustic features, and computes emotion probabilities.
"""

from __future__ import annotations

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import joblib

from feature_extraction import (
    extract_acoustic_features,
    load_audio,
    EMOTION_LABELS,
    get_feature_names,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "emotion_model.pkl"

# Zero-emoji minimal indicator mapping (replaces legacy emoji dictionaries)
EMOTION_ICONS: Dict[str, str] = {
    "Neutral": "",
    "Calm": "",
    "Happy": "",
    "Sad": "",
    "Angry": "",
    "Fearful": "",
    "Disgust": "",
    "Surprised": "",
}

EMOTION_DISCLAIMER: str = (
    "Model output represents an acoustic classification score (estimated probability) "
    "based on vocal characteristics, not a clinical determination of genuine internal emotional state."
)


class EmotionDetector:
    """
    Encapsulates trained Gradient Boosting and Random Forest models for SER inference.
    """

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.bundle: Optional[Dict[str, Any]] = None
        self.scaler = None
        self.classifiers: Dict[str, Any] = {}
        self.classes: List[str] = []
        self.feature_names: List[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Loads serialized model artifact from disk."""
        if not self.model_path.exists():
            return

        try:
            artifact = joblib.load(self.model_path)
            if isinstance(artifact, dict) and "scaler" in artifact:
                self.bundle = artifact
                self.scaler = artifact["scaler"]
                self.classes = list(artifact.get("classes", EMOTION_LABELS))
                self.feature_names = artifact.get("feature_names", get_feature_names())

                if "classifiers" in artifact:
                    self.classifiers = artifact["classifiers"]
                elif "classifier" in artifact:
                    self.classifiers = {"Primary": artifact["classifier"]}
            else:
                raise ValueError("Model artifact must contain 'scaler' and 'classifier'.")
        except Exception as e:
            raise RuntimeError(f"Failed to load emotion model from '{self.model_path}': {e}")

    @property
    def is_loaded(self) -> bool:
        return self.scaler is not None and len(self.classifiers) > 0

    def get_available_models(self) -> List[str]:
        return list(self.classifiers.keys())

    def predict(
        self,
        audio_source: Any,
        model_name: Optional[str] = None,
        target_sr: int = 16000,
    ) -> Dict[str, Any]:
        """
        Runs emotion recognition using the selected model.
        """
        if not self.is_loaded:
            raise FileNotFoundError(
                f"Emotion model artifact not found at '{self.model_path}'. "
                "Please run: python train_emotion_model.py"
            )

        # Select classifier
        if model_name and model_name in self.classifiers:
            clf = self.classifiers[model_name]
            active_model_name = model_name
        else:
            active_model_name = list(self.classifiers.keys())[0]
            clf = self.classifiers[active_model_name]

        # 1. Preprocess and extract features
        y, sr = load_audio(audio_source, target_sr=target_sr)

        # Guard: Silent audio
        if len(y) == 0 or np.max(np.abs(y)) < 1e-5:
            return {
                "status": "warning",
                "message": "Audio is silent or empty. Unable to extract meaningful acoustic features.",
                "predicted_emotion": "Neutral",
                "confidence_score": 0.0,
                "probabilities": {emo: 0.125 for emo in EMOTION_LABELS},
                "icon": EMOTION_ICONS.get("Neutral", ""),
                "waveform": y,
                "sr": sr,
                "model_name": active_model_name,
                "disclaimer": EMOTION_DISCLAIMER,
            }

        feature_vec = extract_acoustic_features(y, sr=sr)
        scaled_features = self.scaler.transform(feature_vec.reshape(1, -1))

        # 2. Predict emotion class
        pred_label = clf.predict(scaled_features)[0]

        # 3. Predict class probabilities
        if hasattr(clf, "predict_proba"):
            probs_raw = clf.predict_proba(scaled_features)[0]
            classes = list(clf.classes_)
            prob_dict = {
                cls_name: float(probs_raw[classes.index(cls_name)]) if cls_name in classes else 0.0
                for cls_name in EMOTION_LABELS
            }
            confidence_score = float(np.max(probs_raw))
        else:
            prob_dict = {emo: (1.0 if emo == pred_label else 0.0) for emo in EMOTION_LABELS}
            confidence_score = 1.0

        return {
            "status": "success",
            "predicted_emotion": pred_label,
            "confidence_score": confidence_score,
            "probabilities": prob_dict,
            "icon": EMOTION_ICONS.get(pred_label, ""),
            "model_name": active_model_name,
            "feature_vector": feature_vec,
            "feature_names": self.feature_names,
            "waveform": y,
            "sr": sr,
            "disclaimer": EMOTION_DISCLAIMER,
        }


_EMOTION_DETECTOR: Optional[EmotionDetector] = None


def get_emotion_detector(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> EmotionDetector:
    global _EMOTION_DETECTOR
    if _EMOTION_DETECTOR is None or str(_EMOTION_DETECTOR.model_path) != str(model_path):
        _EMOTION_DETECTOR = EmotionDetector(model_path=model_path)
    return _EMOTION_DETECTOR


def predict_emotion(
    audio_source: Any,
    model_name: Optional[str] = None,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    detector = get_emotion_detector(model_path=model_path)
    return detector.predict(audio_source, model_name=model_name)
