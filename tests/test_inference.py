"""
SpeechSense AI - End-to-End Pipeline & Inference Tests
======================================================
Tests model loading, ASR transcription (Whisper & Wav2Vec2),
SER emotion prediction (Gradient Boosting & Random Forest),
and dual pipeline execution on identical audio inputs.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pytest

from feature_extraction import load_audio, extract_acoustic_features
from speech_recognition import (
    get_speech_recognizer,
    transcribe_speech,
    MODEL_WHISPER,
    MODEL_WAV2VEC2,
)
from emotion_detection import get_emotion_detector, predict_emotion, EMOTION_LABELS
SAMPLE_AUDIO_DIR = BASE_DIR / "sample_audio"
MODEL_FILE = BASE_DIR / "models" / "emotion_model.pkl"


class TestDualPipelineInference:

    @pytest.fixture
    def sample_audio_path(self):
        sample_file = SAMPLE_AUDIO_DIR / "demo_01_neutral.wav"
        if not sample_file.exists():
            wavs = list(SAMPLE_AUDIO_DIR.glob("*.wav"))
            assert len(wavs) > 0, "No audio files found in sample_audio"
            sample_file = wavs[0]
        return sample_file

    def test_emotion_model_loading(self):
        detector = get_emotion_detector(model_path=MODEL_FILE)
        assert detector.is_loaded is True
        assert detector.scaler is not None
        assert len(detector.classes) == 8
        assert "Gradient Boosting (Latest)" in detector.get_available_models()
        assert "Random Forest (Baseline)" in detector.get_available_models()

    def test_gradient_boosting_emotion_prediction(self, sample_audio_path):
        res = predict_emotion(sample_audio_path, model_name="Gradient Boosting (Latest)")
        assert res["status"] == "success"
        assert res["predicted_emotion"] in EMOTION_LABELS
        assert 0.0 <= res["confidence_score"] <= 1.0
        assert len(res["probabilities"]) == 8
        prob_sum = sum(res["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-4

    def test_random_forest_emotion_prediction(self, sample_audio_path):
        res = predict_emotion(sample_audio_path, model_name="Random Forest (Baseline)")
        assert res["status"] == "success"
        assert res["predicted_emotion"] in EMOTION_LABELS
        assert 0.0 <= res["confidence_score"] <= 1.0
        assert len(res["probabilities"]) == 8
        prob_sum = sum(res["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-4

    def test_whisper_transcription(self, sample_audio_path):
        res = transcribe_speech(sample_audio_path, model_type=MODEL_WHISPER)
        assert res["status"] == "success"
        assert isinstance(res["transcript"], str)
        assert len(res["transcript"]) > 0
        assert res["inference_time_sec"] >= 0.0
        assert "whisper" in res["model_name"].lower()

    def test_wav2vec2_transcription(self, sample_audio_path):
        res = transcribe_speech(sample_audio_path, model_type=MODEL_WAV2VEC2)
        assert res["status"] == "success"
        assert isinstance(res["transcript"], str)
        assert len(res["transcript"]) > 0
        assert "wav2vec2" in res["model_name"].lower()

    def test_dual_pipeline_same_audio(self, sample_audio_path):
        with open(sample_audio_path, "rb") as f:
            audio_bytes = f.read()

        asr_res = transcribe_speech(io.BytesIO(audio_bytes), model_type=MODEL_WHISPER)
        ser_res = predict_emotion(io.BytesIO(audio_bytes), model_name="Gradient Boosting (Latest)")

        assert asr_res["status"] == "success"
        assert ser_res["status"] == "success"
        assert asr_res["transcript"] is not None
        assert ser_res["predicted_emotion"] in EMOTION_LABELS

    def test_silence_guard(self):
        # 1.0 second of absolute silence
        silence_wave = np.zeros(16000, dtype=np.float32)
        ser_res = predict_emotion(silence_wave)
        assert ser_res["status"] == "warning"

        asr_res = transcribe_speech(silence_wave)
        assert asr_res["status"] == "warning"

    def test_actor_independent_split_integrity(self):
        from feature_extraction import scan_ravdess_dataset
        samples = scan_ravdess_dataset(BASE_DIR / "dataset" / "RAVDESS")
        if samples:
            train_actors = set(s["actor_id"] for s in samples if s["actor_id"] <= 20)
            test_actors = set(s["actor_id"] for s in samples if s["actor_id"] >= 21)
            assert len(train_actors.intersection(test_actors)) == 0

    def test_asr_evaluation_metrics_function(self):
        from evaluate_asr import evaluate_asr_transcripts
        refs = ["KIDS ARE TALKING BY THE DOOR", "DOGS ARE SITTING BY THE DOOR"]
        hyps = ["KIDS ARE TALKING BY THE DOOR", "DOGS ARE SITTING BY THE DOOR"]
        res = evaluate_asr_transcripts(refs, hyps)
        assert res["word_error_rate_wer"] == 0.0
        assert res["accuracy_word_level"] == 100.0

        # Test intentional substitution
        hyps_sub = ["KITS ARE TALKING BY THE DOOR", "DOGS ARE SITTING BY THE DOOR"]
        res_sub = evaluate_asr_transcripts(refs, hyps_sub)
        assert res_sub["word_error_rate_wer"] > 0.0


class TestStreamlitAppIntegrity:
    def test_app_compilation(self):
        app_path = BASE_DIR / "app.py"
        assert app_path.exists()
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()
        compiled = compile(source, str(app_path), "exec")
        assert compiled is not None
