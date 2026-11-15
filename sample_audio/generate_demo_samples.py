"""
SpeechSense AI - Demonstration Audio Generator
==============================================
Synthesizes demonstration speech-like WAV files with varied acoustic properties
(pitch F0, formant resonance, and RMS energy envelopes) for testing and live demonstrations.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf

OUTPUT_DIR = Path(__file__).resolve().parent

SR = 16000  # 16 kHz sample rate


def generate_speech_like_signal(
    duration: float = 2.5,
    base_f0: float = 140.0,
    f0_variation: float = 25.0,
    f0_rate: float = 3.0,
    energy_peaks: int = 5,
    brightness: float = 1.0,
) -> np.ndarray:
    """
    Generates a harmonic signal with vowel-like formants and syllabic amplitude envelopes.
    """
    n_samples = int(SR * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # 1. Pitch Contour with intonation modulation
    f0_contour = base_f0 + f0_variation * np.sin(2 * np.pi * f0_rate * t)

    # Accumulate phase
    phase = 2 * np.pi * np.cumsum(f0_contour) / SR

    # 2. Harmonics with formant filtering
    # Formants: F1 ~ 500Hz, F2 ~ 1500Hz, F3 ~ 2500Hz
    signal = np.zeros(n_samples, dtype=np.float32)
    for harmonic_idx in range(1, 30):
        harm_freq = harmonic_idx * base_f0
        if harm_freq >= (SR / 2):
            break
        # Simple acoustic decay
        weight = 1.0 / (harmonic_idx ** 1.1)

        # Formant resonance boost
        f1_boost = np.exp(-((harm_freq - 500) ** 2) / (2 * 120 ** 2))
        f2_boost = np.exp(-((harm_freq - 1500) ** 2) / (2 * 180 ** 2)) * brightness
        f3_boost = np.exp(-((harm_freq - 2500) ** 2) / (2 * 250 ** 2)) * brightness

        amp = weight * (1.0 + 2.5 * f1_boost + 1.8 * f2_boost + 1.2 * f3_boost)
        signal += amp * np.sin(harmonic_idx * phase)

    # 3. Syllabic Envelope (speech cadence)
    envelope = np.zeros(n_samples, dtype=np.float32)
    syllable_times = np.linspace(0.2, duration - 0.2, energy_peaks)
    for st in syllable_times:
        envelope += np.exp(-((t - st) ** 2) / (2 * 0.08 ** 2))

    # Add gentle noise floor
    noise = np.random.normal(0, 0.015, n_samples).astype(np.float32)
    audio = (signal * envelope) + (noise * envelope)

    # Peak normalize
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-4:
        audio = (audio / max_amp) * 0.9

    return audio.astype(np.float32)


def create_demo_suite():
    """Generates 5 distinct demonstration audio clips."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    demos = [
        {
            "name": "sample_01_kids_talking.wav",
            "desc": "Kids are talking by the door (Neutral/Moderate)",
            "f0": 130.0,
            "f0_var": 20.0,
            "dur": 2.8,
            "peaks": 6,
            "brightness": 1.0,
        },
        {
            "name": "sample_02_dogs_sitting.wav",
            "desc": "Dogs are sitting by the door (Calm/Measured)",
            "f0": 115.0,
            "f0_var": 12.0,
            "dur": 3.0,
            "peaks": 6,
            "brightness": 0.8,
        },
        {
            "name": "sample_03_happy_cadence.wav",
            "desc": "Joyful / Animated Speech (High F0, Dynamic)",
            "f0": 210.0,
            "f0_var": 50.0,
            "dur": 2.4,
            "peaks": 7,
            "brightness": 1.4,
        },
        {
            "name": "sample_04_angry_cadence.wav",
            "desc": "Intense / Forceful Speech (High Energy, Sharp)",
            "f0": 195.0,
            "f0_var": 35.0,
            "dur": 2.2,
            "peaks": 5,
            "brightness": 1.6,
        },
        {
            "name": "sample_05_sad_cadence.wav",
            "desc": "Somber / Subdued Speech (Low F0, Slower)",
            "f0": 105.0,
            "f0_var": 8.0,
            "dur": 3.2,
            "peaks": 5,
            "brightness": 0.6,
        },
    ]

    print("[+] Synthesizing sample audio files for demonstration...")
    for item in demos:
        out_path = OUTPUT_DIR / item["name"]
        audio = generate_speech_like_signal(
            duration=item["dur"],
            base_f0=item["f0"],
            f0_variation=item["f0_var"],
            energy_peaks=item["peaks"],
            brightness=item["brightness"],
        )
        sf.write(str(out_path), audio, SR)
        print(f"    Created: {item['name']} ({item['desc']})")

    print("[✓] All sample audio files successfully generated in sample_audio/!\n")


if __name__ == "__main__":
    create_demo_suite()

