"""
SpeechSense AI - ASR Word Error Rate (WER) & Character Error Rate (CER) Evaluation
=================================================================================
Evaluates speech-to-text accuracy of OpenAI Whisper Base and Facebook Wav2Vec 2.0
against ground truth reference transcripts on speech recordings from the RAVDESS corpus.

Evaluation Metric:
- Word Error Rate (WER): (Substitutions + Deletions + Insertions) / Total Words
- Character Error Rate (CER): Levenshtein distance on characters / Total Characters
Computed via the JiWER library with standard text normalization (lowercased, stripped punctuation).
"""

from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import jiwer

from feature_extraction import parse_ravdess_filename, STATEMENT_MAP
from speech_recognition import get_speech_recognizer, MODEL_WHISPER, MODEL_WAV2VEC2

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "RAVDESS"
MODELS_DIR = BASE_DIR / "models"
ASR_REPORT_PATH = MODELS_DIR / "asr_evaluation_report.json"


def select_evaluation_subset(dataset_dir: Path = DATASET_DIR, n_samples: int = 16) -> List[Path]:
    """
    Selects a stratified subset of genuine RAVDESS speech recordings across actors
    and both standardized statements (Kids are talking / Dogs are sitting).
    """
    selected = []
    # Iterate across actors 1 through 16 to get multi-speaker representation
    for aid in range(1, n_samples + 1):
        actor_dir = dataset_dir / f"Actor_{aid:02d}"
        if actor_dir.exists():
            wavs = sorted(actor_dir.glob("*.wav"))
            if wavs:
                # Pick a sample that alternates between Statement 1 and Statement 2
                idx = aid % len(wavs)
                selected.append(wavs[idx])
    return selected


def evaluate_asr_transcripts(
    references: List[str],
    hypotheses: List[str],
    model_name: str = "openai/whisper-base",
) -> Dict[str, Any]:
    """
    Computes standard Word Error Rate (WER) and Character Error Rate (CER) via JiWER.
    """
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
    ])

    clean_refs = transform(references)
    clean_hyps = transform(hypotheses)

    wer = float(jiwer.wer(clean_refs, clean_hyps))
    cer = float(jiwer.cer(clean_refs, clean_hyps))
    mer = float(jiwer.mer(clean_refs, clean_hyps))
    wil = float(jiwer.wil(clean_refs, clean_hyps))

    results = {
        "model": model_name,
        "sample_count": len(references),
        "word_error_rate_wer": round(wer, 4),
        "character_error_rate_cer": round(cer, 4),
        "match_error_rate_mer": round(mer, 4),
        "word_information_lost_wil": round(wil, 4),
        "accuracy_word_level": round((1.0 - wer) * 100, 2),
    }
    return results


def run_asr_evaluation(
    n_samples: int = 16,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    Runs actual model inference on genuine RAVDESS audio recordings,
    computes genuine WER and CER for both Whisper and Wav2Vec 2.0,
    and saves the benchmark report.
    """
    print("=" * 70)
    print("      SPEECHSENSE AI - ASR BENCHMARK EVALUATION (WHISPER & WAV2VEC 2.0)")
    print("=" * 70)

    audio_files = select_evaluation_subset(n_samples=n_samples)
    if not audio_files:
        raise FileNotFoundError(f"No speech recordings found in {DATASET_DIR}")

    print(f"[*] Benchmarking on {len(audio_files)} genuine audio recordings across actors...\n")

    # Load both models
    print("[+] Loading OpenAI Whisper Base...")
    whisp_engine = get_speech_recognizer(MODEL_WHISPER)

    print("[+] Loading Facebook Wav2Vec 2.0 Base...")
    w2v_engine = get_speech_recognizer(MODEL_WAV2VEC2)

    references: List[str] = []
    whisper_hyps: List[str] = []
    wav2vec_hyps: List[str] = []
    eval_details = []

    for f in audio_files:
        meta = parse_ravdess_filename(f)
        ref_text = STATEMENT_MAP.get(meta["statement_code"], "Kids are talking by the door")
        references.append(ref_text)

        t_whisp = whisp_engine.transcribe(f)["transcript"]
        t_w2v = w2v_engine.transcribe(f)["transcript"]

        whisper_hyps.append(t_whisp)
        wav2vec_hyps.append(t_w2v)

        eval_details.append({
            "file": f.name,
            "actor": meta["actor_str"],
            "emotion": meta["emotion"],
            "ground_truth": ref_text,
            "whisper_transcript": t_whisp,
            "wav2vec_transcript": t_w2v,
        })
        print(f"[{f.name}] Ref: '{ref_text}' | Whisper: '{t_whisp}' | Wav2Vec2: '{t_w2v}'")

    # Evaluate both
    whisper_metrics = evaluate_asr_transcripts(references, whisper_hyps, model_name=MODEL_WHISPER)
    wav2vec_metrics = evaluate_asr_transcripts(references, wav2vec_hyps, model_name=MODEL_WAV2VEC2)

    print("\n" + "=" * 70)
    print("                      ASR BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Evaluation Dataset : RAVDESS Speech Subset ({len(audio_files)} multi-speaker files)")
    print("-" * 70)
    print(f"OpenAI Whisper Base:")
    print(f"  Word Error Rate (WER)     : {whisper_metrics['word_error_rate_wer']*100:.2f}%")
    print(f"  Character Error Rate (CER): {whisper_metrics['character_error_rate_cer']*100:.2f}%")
    print(f"  Word-Level Accuracy       : {whisper_metrics['accuracy_word_level']:.2f}%")
    print("-" * 70)
    print(f"Facebook Wav2Vec 2.0 Base:")
    print(f"  Word Error Rate (WER)     : {wav2vec_metrics['word_error_rate_wer']*100:.2f}%")
    print(f"  Character Error Rate (CER): {wav2vec_metrics['character_error_rate_cer']*100:.2f}%")
    print(f"  Word-Level Accuracy       : {wav2vec_metrics['accuracy_word_level']:.2f}%")
    print("=" * 70)

    # Master report payload
    master_report = {
        "dataset": "RAVDESS Speech Corpus",
        "sample_count": len(audio_files),
        "evaluation_scope": (
            f"Stratified {len(audio_files)}-recording evaluation subset across distinct actors and emotions. "
            "Note: RAVDESS contains standardized short read-speech utterances ('Kids are talking by the door' and "
            "'Dogs are sitting by the door'), which results in low error rates compared to spontaneous conversational speech."
        ),
        "models": {
            "OpenAI Whisper Base": whisper_metrics,
            "Facebook Wav2Vec 2.0 Base": wav2vec_metrics,
        },
        # Primary baseline compatibility keys for dashboard display
        "model": MODEL_WHISPER,
        "word_error_rate_wer": whisper_metrics["word_error_rate_wer"],
        "character_error_rate_cer": whisper_metrics["character_error_rate_cer"],
        "accuracy_word_level": whisper_metrics["accuracy_word_level"],
        "details": eval_details,
    }

    if save_report:
        ASR_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ASR_REPORT_PATH, "w", encoding="utf-8") as fp:
            json.dump(master_report, fp, indent=2)
        print(f"\n[+] Saved genuine ASR evaluation metrics to: {ASR_REPORT_PATH.resolve()}")

    return master_report


if __name__ == "__main__":
    run_asr_evaluation()
