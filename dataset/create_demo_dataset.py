"""
SpeechSense AI - Demonstration RAVDESS Dataset Generator
========================================================
Creates a valid RAVDESS speech dataset structure (Actors 01 to 24, Emotions 01 to 08)
with synthesized acoustic speech signals conforming to the official RAVDESS format:
[03]-[01]-[Emotion:01-08]-[Intensity:01-02]-[Statement:01-02]-[Repetition:01-02]-[Actor:01-24].wav

Allows testing the entire training pipeline, actor-independent splitting,
scaler fitting, Random Forest training, and evaluation metrics before/in addition to
downloading the full 1GB RAVDESS archive from Zenodo.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
RAVDESS_DIR = BASE_DIR / "RAVDESS"
SR = 16000

# Emotion acoustic properties (f0, energy, brightness, variation)
EMOTION_ACOUSTIC_PROFILES = {
    "01": {"f0": 130.0, "var": 12.0, "amp": 0.5, "bright": 0.9},   # Neutral
    "02": {"f0": 115.0, "var": 8.0,  "amp": 0.4, "bright": 0.8},   # Calm
    "03": {"f0": 220.0, "var": 45.0, "amp": 0.8, "bright": 1.4},   # Happy
    "04": {"f0": 100.0, "var": 10.0, "amp": 0.35, "bright": 0.6},  # Sad
    "05": {"f0": 210.0, "var": 35.0, "amp": 0.95, "bright": 1.6},  # Angry
    "06": {"f0": 240.0, "var": 50.0, "amp": 0.7, "bright": 1.3},   # Fearful
    "07": {"f0": 120.0, "var": 15.0, "amp": 0.5, "bright": 0.7},   # Disgust
    "08": {"f0": 260.0, "var": 60.0, "amp": 0.85, "bright": 1.5},  # Surprised
}


def synthesize_emotional_speech(
    profile: dict,
    actor_id: int,
    duration: float = 2.0,
) -> np.ndarray:
    """Synthesizes a short speech-like waveform reflecting the emotional acoustic profile."""
    n_samples = int(SR * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Actor pitch shift: female actors (even IDs) have ~40Hz higher fundamental pitch
    pitch_offset = 35.0 if (actor_id % 2 == 0) else 0.0
    base_f0 = profile["f0"] + pitch_offset
    f0_contour = base_f0 + profile["var"] * np.sin(2 * np.pi * 3.5 * t)

    phase = 2 * np.pi * np.cumsum(f0_contour) / SR

    # Harmonic acoustic synthesis
    signal = np.zeros(n_samples, dtype=np.float32)
    for h in range(1, 24):
        freq = h * base_f0
        if freq >= (SR / 2):
            break
        weight = 1.0 / (h ** 1.05)
        # Formants
        f1 = np.exp(-((freq - 550) ** 2) / (2 * 130 ** 2))
        f2 = np.exp(-((freq - 1600) ** 2) / (2 * 200 ** 2)) * profile["bright"]
        amp = weight * (1.0 + 2.0 * f1 + 1.5 * f2)
        signal += amp * np.sin(h * phase)

    # Syllable envelope
    envelope = np.zeros(n_samples, dtype=np.float32)
    syllable_centers = np.linspace(0.2, duration - 0.2, 5)
    for c in syllable_centers:
        envelope += np.exp(-((t - c) ** 2) / (2 * 0.09 ** 2))

    audio = signal * envelope * profile["amp"]
    max_val = np.max(np.abs(audio))
    if max_val > 1e-4:
        audio = (audio / max_val) * 0.85

    return audio.astype(np.float32)


def generate_dataset(num_actors: int = 24):
    """
    Populates dataset/RAVDESS/ with Actor_01 to Actor_{num_actors}
    containing valid speech files for all 8 emotions.
    """
    RAVDESS_DIR.mkdir(parents=True, exist_ok=True)
    total_files = 0

    print(f"[+] Generating demonstration RAVDESS structure across {num_actors} actors...")

    for actor in range(1, num_actors + 1):
        actor_folder = RAVDESS_DIR / f"Actor_{actor:02d}"
        actor_folder.mkdir(parents=True, exist_ok=True)

        for emo_code, profile in EMOTION_ACOUSTIC_PROFILES.items():
            # Generate Statement 1, Repetition 1
            filename = f"03-01-{emo_code}-01-01-01-{actor:02d}.wav"
            filepath = actor_folder / filename

            # Only generate if file doesn't already exist
            if not filepath.exists():
                audio = synthesize_emotional_speech(profile, actor_id=actor)
                sf.write(str(filepath), audio, SR)
                total_files += 1

    print(f"[✓] Created {total_files} RAVDESS demonstration speech files across {num_actors} actors in:")
    print(f"    {RAVDESS_DIR.resolve()}\n")


if __name__ == "__main__":
    generate_dataset()

