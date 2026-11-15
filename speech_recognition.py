"""
SpeechSense AI - Dual ASR Engine Module
========================================
Supports both:
1. OpenAI Whisper Base (Latest & State-of-the-Art Speech-to-Text)
2. Facebook Wav2Vec 2.0 Base (Classical Academic CTC Benchmark)
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Set cache and suppress TF oneDNN warnings before imports
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_NUMBA_DIR = _CACHE_DIR / "numba"
_MPL_DIR = _CACHE_DIR / "matplotlib"
_NUMBA_DIR.mkdir(parents=True, exist_ok=True)
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import numpy as np
import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from feature_extraction import load_audio, TARGET_SAMPLE_RATE

MODEL_WAV2VEC2: str = "facebook/wav2vec2-base-960h"
MODEL_WHISPER: str = "openai/whisper-base"

AVAILABLE_ASR_MODELS = {
    "OpenAI Whisper Base (Primary ASR Model)": MODEL_WHISPER,
    "Facebook Wav2Vec 2.0 Base (Academic Baseline)": MODEL_WAV2VEC2,
}


class SpeechRecognizer:
    """
    Dual-model ASR engine supporting OpenAI Whisper and Wav2Vec 2.0.
    """

    def __init__(
        self,
        model_type: str = MODEL_WHISPER,
        device: Optional[str] = None,
    ):
        self.model_type = model_type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads processor and weights for either Whisper or Wav2Vec 2.0."""
        is_whisper = "whisper" in self.model_type.lower()
        try:
            if is_whisper:
                # Load Whisper from local cache
                self.processor = WhisperProcessor.from_pretrained(self.model_type, local_files_only=True)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_type, local_files_only=True)
            else:
                # Load Wav2Vec2 from local cache
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_type, local_files_only=True)
                self.model = Wav2Vec2ForCTC.from_pretrained(self.model_type, local_files_only=True)
        except Exception:
            # Fall back to online download if not cached
            if is_whisper:
                self.processor = WhisperProcessor.from_pretrained(self.model_type)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_type)
            else:
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_type)
                self.model = Wav2Vec2ForCTC.from_pretrained(self.model_type)

        self.model.to(self.device)
        self.model.eval()

    def transcribe(
        self,
        audio_source: Any,
        target_sr: int = TARGET_SAMPLE_RATE,
    ) -> Dict[str, Any]:
        """
        Transcribes speech audio into clean English text.
        """
        # 1. Preprocess audio
        try:
            y, sr = load_audio(audio_source, target_sr=target_sr)
        except Exception as load_err:
            return {
                "status": "error",
                "message": f"Could not decode audio file: {load_err}",
                "transcript": "",
                "inference_time_sec": 0.0,
                "model_name": self.model_type,
            }

        # Guard: Silence or empty audio (< 0.2s)
        if len(y) < int(sr * 0.2) or np.max(np.abs(y)) < 1e-4:
            return {
                "status": "warning",
                "message": "Audio is silent or too short for speech recognition.",
                "transcript": "(Silence or unintelligible speech detected)",
                "inference_time_sec": 0.0,
                "model_name": self.model_type,
                "waveform": y,
                "sr": sr,
            }

        # 2. Run inference
        t0 = time.time()
        is_whisper = "whisper" in self.model_type.lower()

        try:
            with torch.no_grad():
                if is_whisper:
                    # Whisper feature extraction: Log-mel spectrogram
                    input_features = self.processor(
                        y,
                        sampling_rate=sr,
                        return_tensors="pt",
                    ).input_features.to(self.device)

                    # Autoregressive decoding with repetition penalty and token limit
                    predicted_ids = self.model.generate(
                        input_features,
                        language="en",
                        task="transcribe",
                        max_new_tokens=64,
                        no_repeat_ngram_size=3,
                    )
                    transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                else:
                    # Wav2Vec2 CTC decoding
                    input_values = self.processor(
                        y,
                        sampling_rate=sr,
                        return_tensors="pt",
                        padding=True,
                    ).input_values.to(self.device)

                    logits = self.model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    raw_text = self.processor.batch_decode(predicted_ids)[0].strip()
                    transcript = raw_text[0].upper() + raw_text[1:].lower() if raw_text else ""

            elapsed = round(time.time() - t0, 3)

            return {
                "status": "success",
                "transcript": transcript if transcript else "(No clear words recognized)",
                "inference_time_sec": elapsed,
                "device": str(self.device),
                "model_name": self.model_type,
                "waveform": y,
                "sr": sr,
            }

        except Exception as infer_err:
            return {
                "status": "error",
                "message": f"Inference error: {infer_err}",
                "transcript": "",
                "inference_time_sec": 0.0,
                "model_name": self.model_type,
            }


# Singleton cache by model name
_RECOGNIZERS: Dict[str, SpeechRecognizer] = {}


def get_speech_recognizer(model_type: str = MODEL_WHISPER) -> SpeechRecognizer:
    global _RECOGNIZERS
    if model_type not in _RECOGNIZERS:
        _RECOGNIZERS[model_type] = SpeechRecognizer(model_type=model_type)
    return _RECOGNIZERS[model_type]


def transcribe_speech(audio_source: Any, model_type: str = MODEL_WHISPER) -> Dict[str, Any]:
    recognizer = get_speech_recognizer(model_type=model_type)
    return recognizer.transcribe(audio_source)
