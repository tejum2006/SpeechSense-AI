"""
Unit Tests for RAVDESS Filename Parsing and Acoustic Feature Extraction
======================================================================
"""

import io
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from feature_extraction import (
    parse_ravdess_filename,
    load_audio,
    extract_acoustic_features,
    extract_features_from_audio,
    get_feature_names,
    EMOTION_CODE_MAP,
    EMOTION_LABELS,
    TARGET_SAMPLE_RATE,
)


class TestRAVDESSParsing:
    """Tests for official RAVDESS filename parsing logic."""

    def test_valid_speech_filename(self):
        # 03 = audio-only, 01 = speech, 05 = angry, 01 = normal, 02 = statement 2, 01 = rep 1, 12 = actor 12 (female)
        filename = "03-01-05-01-02-01-12.wav"
        meta = parse_ravdess_filename(filename)

        assert meta["modality"] == "audio-only"
        assert meta["vocal_channel"] == "speech"
        assert meta["is_speech"] is True
        assert meta["emotion_code"] == "05"
        assert meta["emotion"] == "Angry"
        assert meta["intensity"] == "normal"
        assert meta["actor_id"] == 12
        assert meta["gender"] == "female"
        assert meta["actor_str"] == "Actor_12"

    def test_male_actor_odd_id(self):
        filename = "03-01-03-02-01-02-07.wav"
        meta = parse_ravdess_filename(filename)
        assert meta["actor_id"] == 7
        assert meta["gender"] == "male"
        assert meta["emotion"] == "Happy"
        assert meta["intensity"] == "strong"

    def test_all_eight_emotions_mapped(self):
        expected = {
            "01": "Neutral",
            "02": "Calm",
            "03": "Happy",
            "04": "Sad",
            "05": "Angry",
            "06": "Fearful",
            "07": "Disgust",
            "08": "Surprised",
        }
        assert EMOTION_CODE_MAP == expected

        for code, name in expected.items():
            filename = f"03-01-{code}-01-01-01-01.wav"
            meta = parse_ravdess_filename(filename)
            assert meta["emotion"] == name
            assert meta["emotion_code"] == code

    def test_song_modality_detection(self):
        # vocal channel 02 = song
        filename = "03-02-03-01-01-01-04.wav"
        meta = parse_ravdess_filename(filename)
        assert meta["vocal_channel"] == "song"
        assert meta["is_speech"] is False

    def test_invalid_filename_length(self):
        with pytest.raises(ValueError, match="Expected 7 hyphen-separated"):
            parse_ravdess_filename("03-01-05-01-02.wav")

    def test_invalid_emotion_code(self):
        with pytest.raises(ValueError, match="Unknown emotion code"):
            parse_ravdess_filename("03-01-99-01-01-01-01.wav")

    def test_invalid_actor_code(self):
        with pytest.raises(ValueError, match="Invalid actor code"):
            parse_ravdess_filename("03-01-01-01-01-01-XX.wav")


class TestAudioFeatureExtraction:
    """Tests for audio loading and acoustic feature extraction."""

    @pytest.fixture
    def synthetic_tone(self):
        # 1.5 seconds of 220Hz harmonic audio
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        y = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
        return y.astype(np.float32), sr

    @pytest.fixture
    def silent_audio(self):
        sr = 16000
        y = np.zeros(int(sr * 1.0), dtype=np.float32)
        return y, sr

    def test_feature_dimension_and_names(self, synthetic_tone):
        y, sr = synthetic_tone
        feats = extract_acoustic_features(y, sr=sr)
        names = get_feature_names()

        # Dimension should be exactly 61
        assert feats.shape == (61,)
        assert len(names) == 61
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()

    def test_pitch_detected_on_harmonic_tone(self, synthetic_tone):
        y, sr = synthetic_tone
        feats = extract_acoustic_features(y, sr=sr)
        names = get_feature_names()

        # Index of pitch_mean is 52 (after 26 mfcc + 26 delta_mfcc)
        pitch_mean_idx = names.index("pitch_mean")
        pitch_mean_val = feats[pitch_mean_idx]

        # The tone is 220 Hz; detected pitch should be close to 220 Hz (+/- 25 Hz)
        assert 190.0 < pitch_mean_val < 250.0

    def test_silent_audio_safety(self, silent_audio):
        y, sr = silent_audio
        feats = extract_acoustic_features(y, sr=sr)
        assert feats.shape == (61,)
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()
        # Silent audio should produce 0 RMS and 0 Pitch
        assert feats[get_feature_names().index("rms_mean")] == 0.0

    def test_short_audio_handling(self):
        # 20ms of audio (too short for normal analysis)
        y = np.random.randn(int(16000 * 0.02)).astype(np.float32)
        feats = extract_acoustic_features(y, sr=16000)
        assert feats.shape == (61,)
        assert not np.isnan(feats).any()

    def test_stereo_to_mono_and_resampling(self):
        # Stereo 44.1 kHz audio
        sr = 44100
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 300 * t)
        right = 0.3 * np.sin(2 * np.pi * 600 * t)
        stereo = np.column_stack((left, right)).astype(np.float32)

        # Write to in-memory WAV buffer
        buf = io.BytesIO()
        sf.write(buf, stereo, sr, format="WAV")

        # Load via load_audio
        y_mono, target_sr = load_audio(buf, target_sr=16000)

        assert y_mono.ndim == 1
        assert target_sr == 16000
        assert len(y_mono) == 16000

        # Features from buffer
        feats = extract_features_from_audio(buf)
        assert feats.shape == (61,)
        assert not np.isnan(feats).any()

