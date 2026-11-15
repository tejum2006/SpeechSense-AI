"""
SpeechSense AI - Feature Extraction Module
==========================================
Provides reusable acoustic feature extraction and RAVDESS dataset parsing.

Acoustic Features Extracted:
- MFCC (13 coefficients, mean & std across time) -> 26 features
- Delta MFCC (13 coefficients, mean & std across time) -> 26 features
- Fundamental Frequency / Pitch F0 (mean, std, min, max) -> 4 features
- RMS Energy (mean, std, max) -> 3 features
- Spectral Centroid (mean, std) -> 2 features
Total Feature Vector Dimension: 61 features
"""

from __future__ import annotations

import os
import io
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Set writable cache directories for Numba and Matplotlib before importing them
_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_NUMBA_DIR = _CACHE_DIR / "numba"
_MPL_DIR = _CACHE_DIR / "matplotlib"
_NUMBA_DIR.mkdir(parents=True, exist_ok=True)
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import numpy as np
import soundfile as sf
import librosa


# Standard sampling rate across all SER and ASR pipelines
TARGET_SAMPLE_RATE: int = 16000

# Standard 8-class RAVDESS emotion mapping
EMOTION_CODE_MAP: Dict[str, str] = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}

# Emotion order for display and model alignment
EMOTION_LABELS: List[str] = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
]

# Modality and Channel Mappings
MODALITY_MAP: Dict[str, str] = {
    "01": "full-av",
    "02": "video-only",
    "03": "audio-only",
}

VOCAL_CHANNEL_MAP: Dict[str, str] = {
    "01": "speech",
    "02": "song",
}

INTENSITY_MAP: Dict[str, str] = {
    "01": "normal",
    "02": "strong",
}

STATEMENT_MAP: Dict[str, str] = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def parse_ravdess_filename(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parses official RAVDESS audio filename convention.

    Format:
    [Modality]-[VocalChannel]-[Emotion]-[Intensity]-[Statement]-[Repetition]-[Actor].wav
    Example: 03-01-05-01-02-01-12.wav
      03 = audio-only
      01 = speech
      05 = angry
      01 = normal intensity
      02 = statement 2
      01 = 1st repetition
      12 = actor 12 (female)

    Returns:
        dict containing parsed metadata and validated fields.
    Raises:
        ValueError if filename does not conform to the RAVDESS convention.
    """
    path_obj = Path(file_path)
    stem = path_obj.stem

    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"Invalid RAVDESS filename '{path_obj.name}'. "
            f"Expected 7 hyphen-separated numeric parts, found {len(parts)}."
        )

    modality_code, vocal_code, emotion_code, intensity_code, statement_code, rep_code, actor_code = parts

    if emotion_code not in EMOTION_CODE_MAP:
        raise ValueError(f"Unknown emotion code '{emotion_code}' in '{path_obj.name}'. Expected 01-08.")

    try:
        actor_id = int(actor_code)
    except ValueError:
        raise ValueError(f"Invalid actor code '{actor_code}' in '{path_obj.name}'.")

    # Odd = male, Even = female in RAVDESS
    gender = "male" if (actor_id % 2 == 1) else "female"
    is_speech = (vocal_code == "01")

    return {
        "file_path": str(path_obj.resolve()),
        "filename": path_obj.name,
        "modality_code": modality_code,
        "modality": MODALITY_MAP.get(modality_code, "unknown"),
        "vocal_channel_code": vocal_code,
        "vocal_channel": VOCAL_CHANNEL_MAP.get(vocal_code, "unknown"),
        "is_speech": is_speech,
        "emotion_code": emotion_code,
        "emotion": EMOTION_CODE_MAP[emotion_code],
        "intensity_code": intensity_code,
        "intensity": INTENSITY_MAP.get(intensity_code, "unknown"),
        "statement_code": statement_code,
        "statement": STATEMENT_MAP.get(statement_code, "unknown"),
        "repetition_code": rep_code,
        "actor_id": actor_id,
        "actor_str": f"Actor_{actor_id:02d}",
        "gender": gender,
    }


def load_audio(
    audio_source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Loads, mono-converts, and resamples an audio input.

    Supports:
    - File path string or Path
    - Raw audio bytes or BytesIO buffer
    - Existing NumPy float waveform

    Returns:
        (y, sr): Mono 1D float32 numpy array and sampling rate.
    """
    if isinstance(audio_source, np.ndarray):
        y = audio_source.astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)
        return y, target_sr

    if isinstance(audio_source, (bytes, io.BytesIO)):
        buf = io.BytesIO(audio_source) if isinstance(audio_source, bytes) else audio_source
        buf.seek(0)
        y, orig_sr = sf.read(buf, dtype="float32")
    else:
        file_path = Path(audio_source)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        y, orig_sr = sf.read(str(file_path), dtype="float32")

    # Convert to mono if multi-channel
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample to target_sr if required
    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

    # Normalize audio amplitude to [-1.0, 1.0] if not silent
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val

    return y.astype(np.float32), target_sr


def get_feature_names(n_mfcc: int = 13) -> List[str]:
    """
    Returns ordered names for all 61 extracted acoustic features.
    """
    names: List[str] = []
    # 1. MFCC Means
    for i in range(1, n_mfcc + 1):
        names.append(f"mfcc_{i}_mean")
    # 2. MFCC Stds
    for i in range(1, n_mfcc + 1):
        names.append(f"mfcc_{i}_std")
    # 3. Delta MFCC Means
    for i in range(1, n_mfcc + 1):
        names.append(f"delta_mfcc_{i}_mean")
    # 4. Delta MFCC Stds
    for i in range(1, n_mfcc + 1):
        names.append(f"delta_mfcc_{i}_std")
    # 5. Pitch F0
    names.extend(["pitch_mean", "pitch_std", "pitch_min", "pitch_max"])
    # 6. RMS Energy
    names.extend(["rms_mean", "rms_std", "rms_max"])
    # 7. Spectral Centroid
    names.extend(["spectral_centroid_mean", "spectral_centroid_std"])
    return names


def extract_acoustic_features(
    y: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Extracts a fixed-length 1D acoustic feature vector (length = 61) from audio waveform.

    Components:
    - MFCC: 13 mean + 13 std = 26
    - Delta MFCC: 13 mean + 13 std = 26
    - Pitch (F0): mean, std, min, max = 4
    - RMS Energy: mean, std, max = 3
    - Spectral Centroid: mean, std = 2
    Total: 61 features

    Safe guards:
    - Handles silence and near-silent signals gracefully.
    - Replaces all NaNs and infinities with 0.0.
    """
    feature_vector: List[float] = []

    # Safeguard: Empty or extremely short audio (< 0.1s)
    if y is None or len(y) < int(sr * 0.1) or np.max(np.abs(y)) < 1e-5:
        # Return all zeros vector of expected dimension
        expected_dim = (n_mfcc * 4) + 4 + 3 + 2
        return np.zeros(expected_dim, dtype=np.float32)

    # 1. MFCC Features (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feature_vector.extend(mfcc_mean.tolist())
    feature_vector.extend(mfcc_std.tolist())

    # 2. Delta MFCC Features
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    feature_vector.extend(delta_mean.tolist())
    feature_vector.extend(delta_std.tolist())

    # 3. Fundamental Frequency / Pitch (F0)
    # Using librosa.yin for robust, fast speech fundamental frequency tracking (50 Hz - 500 Hz)
    try:
        f0 = librosa.yin(
            y,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=2048,
            hop_length=512,
            trough_threshold=0.1,
        )
        # Filter unvoiced or boundary artifacts (50 Hz or 500 Hz limits)
        valid_f0 = f0[(f0 > 55.0) & (f0 < 490.0) & (~np.isnan(f0))]
        if len(valid_f0) > 0:
            pitch_mean = float(np.mean(valid_f0))
            pitch_std = float(np.std(valid_f0))
            pitch_min = float(np.min(valid_f0))
            pitch_max = float(np.max(valid_f0))
        else:
            pitch_mean, pitch_std, pitch_min, pitch_max = 0.0, 0.0, 0.0, 0.0
    except Exception:
        pitch_mean, pitch_std, pitch_min, pitch_max = 0.0, 0.0, 0.0, 0.0

    feature_vector.extend([pitch_mean, pitch_std, pitch_min, pitch_max])

    # 4. RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms))
    feature_vector.extend([rms_mean, rms_std, rms_max])

    # 5. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = float(np.mean(spectral_centroid))
    sc_std = float(np.std(spectral_centroid))
    feature_vector.extend([sc_mean, sc_std])

    # Convert to numpy array and sanitize against NaN/inf
    arr = np.array(feature_vector, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def extract_features_from_audio(
    audio_source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    End-to-end convenience wrapper: loads audio and extracts the 61-dim feature vector.
    """
    y, sr = load_audio(audio_source, target_sr=target_sr)
    return extract_acoustic_features(y, sr=sr)


def scan_ravdess_dataset(
    dataset_dir: Union[str, Path],
    speech_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Scans the RAVDESS dataset directory recursively for WAV files.
    Parses filenames and returns a list of sample records.

    Args:
        dataset_dir: Root path to RAVDESS dataset (containing Actor_* subfolders or files)
        speech_only: If True, filters strictly for speech recordings (vocal_channel == '01')
    """
    root = Path(dataset_dir)
    if not root.exists():
        return []

    samples: List[Dict[str, Any]] = []
    # Search all .wav files recursively
    for wav_file in root.rglob("*.wav"):
        try:
            meta = parse_ravdess_filename(wav_file)
            if speech_only and not meta["is_speech"]:
                continue
            samples.append(meta)
        except ValueError:
            # Skip non-RAVDESS or improperly named files
            continue

    return samples

