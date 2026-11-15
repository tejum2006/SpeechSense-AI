# SpeechSense AI: Complete Technical Audit & Verification Report

**Project Title:** SpeechSense AI – Dual-Pipeline Speech Recognition and Vocal Emotion Analysis System  
**Audit Conducted by:** Senior Machine Learning Engineer, Speech Processing Researcher, & SQA Lead  
**Audit Date:** 2026-11-11  
**Target Environment:** Windows 11, Python 3.10.11, PyTorch 2.5.1 (CPU), Transformers 4.40+, Scikit-Learn 1.3+  

---

## 1. Executive Summary

A comprehensive, end-to-end technical audit and empirical verification of **SpeechSense AI** was conducted across all subsystems: data ingestion, acoustic feature engineering, deep learning Automatic Speech Recognition (ASR), supervised Speech Emotion Recognition (SER), experimental methodology, automated test suites, and the Streamlit web dashboard.

### Key Audit Findings
1. **Zero Mock or Hardcoded Logic:** All predictions, transcriptions, and metrics are computed dynamically from actual audio waveforms.
2. **Actor-Independent Integrity:** Strict speaker separation on the official RAVDESS corpus (Actors 01–20 in Train, Actors 21–24 in Test) was verified with mathematically confirmed **zero speaker leakage**.
3. **Genuine Empirical Metrics:** 
   - SER Primary Classifier (`HistGradientBoostingClassifier`): **43.75% 8-class accuracy** (~3.5x higher than random chance of 12.5%).
   - SER Baseline Classifier (`RandomForestClassifier`): **38.75% 8-class accuracy**.
   - ASR Primary Engine (`openai/whisper-base`): **1.04% WER / 0.89% CER** on multi-speaker RAVDESS evaluation.
   - ASR Baseline Engine (`facebook/wav2vec2-base-960h`): **2.08% WER / 0.45% CER** on multi-speaker RAVDESS evaluation.
4. **Defect Remediation:** Corrected an artificial mock comparison in `evaluate_asr.py` by implementing genuine audio-based JiWER evaluation against official ground truth statements; corrected Plotly axis font property incompatibilities in `app.py`; expanded test suite to 22 passing tests.

---

## 2. Existing Project Inventory

| Component | File / Path | Implementation Status | Origin / Training Details |
| :--- | :--- | :---: | :--- |
| **ASR Primary Engine** | `speech_recognition.py` | Verified & Operational | Pretrained `openai/whisper-base` (74M params) via Hugging Face Transformers |
| **ASR Baseline Engine** | `speech_recognition.py` | Verified & Operational | Pretrained `facebook/wav2vec2-base-960h` (95M params) via Hugging Face Transformers |
| **SER Primary Model** | `models/emotion_model.pkl` | Verified & Operational | Trained locally on 1,200 RAVDESS files (`HistGradientBoostingClassifier`) |
| **SER Baseline Model** | `models/emotion_model.pkl` | Verified & Operational | Trained locally on 1,200 RAVDESS files (`RandomForestClassifier`, 150 trees) |
| **Feature Scaler** | `models/emotion_model.pkl` | Verified & Operational | `StandardScaler` fitted **exclusively** on training set (Actors 01–20) |
| **Feature Extraction** | `feature_extraction.py` | Verified & Operational | 61 acoustic descriptors (MFCCs, Deltas, Pitch F0, RMS, Spectral Centroid) |
| **Web Dashboard** | `app.py` | Verified & Operational | Modern minimal Streamlit interface with zero emojis and responsive Plotly charts |
| **Dataset Corpus** | `dataset/RAVDESS/` | Verified & Complete | 1,440 genuine human speech recordings across all 24 professional actors |
| **Demo Audio Samples** | `sample_audio/` | Verified & Genuine | 8 genuine RAVDESS human speech WAV files covering all 8 emotion classes |
| **SER Evaluation** | `evaluate_emotion_model.py` | Verified & Operational | Generates evaluation report and `assets/confusion_matrix.png` |
| **ASR Evaluation** | `evaluate_asr.py` | Verified & Remediated | Dynamic multi-speaker JiWER evaluation on genuine audio files |
| **Automated Tests** | `tests/` | Verified (22 Passed) | Unit tests in `test_feature_extraction.py` and integration in `test_inference.py` |

---

## 3. Architecture & Data Flow Verification

```
                             Raw Audio Input (.wav)
                                       │
                        Mono Resampling @ 16,000 Hz
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
   ASR Pipeline (Transformers)             SER Pipeline (Acoustic Descriptors)
   ├─ Whisper: Log-mel spectrogram         ├─ 26 MFCCs (13 Mean, 13 Std)
   │  80 channels -> Encoder-Decoder       ├─ 26 Delta MFCCs (13 Mean, 13 Std)
   └─ Wav2Vec2: 1D normalized waveform     ├─ 4 Pitch F0 (Mean, Std, Min, Max)
      Conv Encoder -> CTC Argmax Decode    ├─ 3 RMS Energy (Mean, Std, Max)
                  │                        └─ 2 Spectral Centroid (Mean, Std)
                  ▼                                         │
        Spoken Text Transcript                              ▼
   "Kids are talking by the door."               StandardScaler Transform
                                                            │
                                                            ▼
                                               Histogram Gradient Boosting /
                                                  Random Forest Classifier
                                                            │
                                                            ▼
                                                 Predicted Emotion & Probs
                                                    ANGRY (87.5% Prob)
```

### Separation of ASR Processing Pipelines
- **OpenAI Whisper Base:** Utilizes `WhisperProcessor` to compute an 80-channel log-mel spectrogram over 3000-frame windows. Generates autoregressive tokens using `model.generate(language="en", task="transcribe", max_new_tokens=64, no_repeat_ngram_size=3)`.
- **Facebook Wav2Vec 2.0:** Utilizes `Wav2Vec2Processor` to pass normalized 1D raw waveform arrays into a 7-layer convolutional feature encoder followed by a 12-layer Transformer context network and a linear CTC head (`torch.argmax(logits, dim=-1)`).
- **Audit Verdict:** The two models use strictly separated and architecturally correct preprocessing pipelines. Neither pipeline leaks or improperly applies the other's feature representation.

---

## 4. Dataset Verification & Validation

The official speech partition of the RAVDESS dataset was audited directly on the local filesystem:

- **Root Directory:** `dataset/RAVDESS/`
- **Actor Folders:** 24 distinct directories (`Actor_01` through `Actor_24`).
- **Total Speech Files:** Exactly **1,440 files** (verified: `len(glob.glob('dataset/RAVDESS/**/*.wav')) == 1440`).
- **Files per Actor:** Exactly **60 files per actor** without exception.
- **Audio Integrity:** 0 corrupted files, 0 zero-length frames, 0 unreadable headers.
- **Duration Statistics:** Minimum = 2.94s, Maximum = 5.27s, Mean = 3.70s.
- **Class Distribution:**
  - Neutral: 96 samples (4 per actor * 24 actors; neutral has only normal intensity in RAVDESS).
  - Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised: 192 samples each (8 per actor * 24 actors).
  - Total: $96 + (7 \times 192) = 1,440$ samples.

---

## 5. Acoustic Feature Extraction Verification (61 Dimensions)

The feature extractor [`feature_extraction.py`](file:///c:/Users/DELL/OneDrive/Desktop/Speech%20Emotion/feature_extraction.py) was verified against mathematical and dimension specifications:

$$\mathbf{x} = [\mu_{\text{mfcc}}^{1:13},\, \sigma_{\text{mfcc}}^{1:13},\, \mu_{\Delta\text{mfcc}}^{1:13},\, \sigma_{\Delta\text{mfcc}}^{1:13},\, \mu_{F_0},\, \sigma_{F_0},\, \min_{F_0},\, \max_{F_0},\, \mu_{\text{rms}},\, \sigma_{\text{rms}},\, \max_{\text{rms}},\, \mu_{\text{sc}},\, \sigma_{\text{sc}}] \in \mathbb{R}^{61}$$

### Dimensionality Audit
| Category | Descriptors | Dimensions | Verified Extraction Function |
| :--- | :--- | :---: | :--- |
| **MFCC Means** | Coefficients 1–13 temporal means | 13 | `librosa.feature.mfcc(y, sr, n_mfcc=13).mean(axis=1)` |
| **MFCC Standard Deviations** | Coefficients 1–13 temporal variances | 13 | `librosa.feature.mfcc(y, sr, n_mfcc=13).std(axis=1)` |
| **Delta MFCC Means** | First-order temporal derivative means | 13 | `librosa.feature.delta(mfcc).mean(axis=1)` |
| **Delta MFCC Stds** | First-order temporal derivative stds | 13 | `librosa.feature.delta(mfcc).std(axis=1)` |
| **Pitch ($F_0$) Statistics** | Voiced fundamental frequency (Mean, Std, Min, Max) | 4 | `librosa.yin(y, fmin=50, fmax=500)` with unvoiced filtering |
| **RMS Energy Statistics** | Amplitude envelope (Mean, Std, Max) | 3 | `librosa.feature.rms(y)` |
| **Spectral Centroid** | Spectral brightness center (Mean, Std) | 2 | `librosa.feature.spectral_centroid(y, sr)` |
| **Total Dimension** | — | **61** | Verified: Output vector shape is strictly `(61,)` |

### Robustness & Safeguards
- **Silence & Short Audio:** Verified to return an all-zero vector of shape `(61,)` when input duration is $< 0.1\text{ s}$ or maximum amplitude is $< 10^{-5}$.
- **NaN / Infinity Sanitation:** Every vector is post-processed through `np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)`.
- **Scaling Consistency:** Verified that `StandardScaler` was fitted exclusively on the 1,200 training samples and serialized in `emotion_model.pkl`. The inference module loads this identical transformer.

---

## 6. Experimental Methodology & SER Evaluation

### 6.1 Actor-Independent Partition Audit
- **Training Partition:** Actors 01–20 (1,200 speech recordings across 10 male and 10 female speakers).
- **Testing Partition:** Actors 21–24 (240 speech recordings across 2 male and 2 female speakers).
- **Leakage Verification:** $\text{Actors}_{\text{train}} \cap \text{Actors}_{\text{test}} = \emptyset$. Confirmed zero identity leakage.

### 6.2 Empirical Results on Unseen Actors (Actors 21–24)

Both models were evaluated on the 240 held-out test recordings:

```
======================================================================
Model: Gradient Boosting (HistGradientBoostingClassifier - Primary)
Accuracy:        43.75%
Macro Precision: 44.19%
Macro Recall:    42.58%
Macro F1-Score:  41.65%
----------------------------------------------------------------------
Classification Report:
              precision    recall  f1-score   support
     Neutral       0.27      0.25      0.26        16
        Calm       0.59      0.31      0.41        32
       Happy       0.50      0.25      0.33        32
         Sad       0.23      0.22      0.23        32
       Angry       0.40      0.66      0.49        32
     Fearful       0.63      0.53      0.58        32
     Disgust       0.42      0.50      0.46        32
   Surprised       0.50      0.69      0.58        32
    accuracy                           0.44       240
   macro avg       0.44      0.43      0.42       240
weighted avg       0.45      0.44      0.43       240

Confusion Matrix:
[[ 4  0  2  7  0  0  2  1]   <- Neutral
 [ 3 10  1  7  0  2  3  6]   <- Calm
 [ 1  1  8  2  5  1  4 10]   <- Happy
 [ 6  6  2  7  2  1  7  1]   <- Sad
 [ 0  0  2  1 21  4  3  1]   <- Angry
 [ 0  0  0  5  5 17  3  2]   <- Fearful
 [ 1  0  0  1 12  1 16  1]   <- Disgust
 [ 0  0  1  0  8  1  0 22]]  <- Surprised
```

```
======================================================================
Model: Random Forest (RandomForestClassifier, 150 Trees - Baseline)
Accuracy:        38.75%
Macro Precision: 38.30%
Macro Recall:    37.89%
Macro F1-Score:  36.47%
----------------------------------------------------------------------
Classification Report:
              precision    recall  f1-score   support
     Neutral       0.40      0.25      0.31        16
        Calm       0.64      0.56      0.60        32
       Happy       0.32      0.19      0.24        32
         Sad       0.24      0.12      0.16        32
       Angry       0.36      0.78      0.50        32
     Fearful       0.55      0.56      0.55        32
     Disgust       0.25      0.25      0.25        32
   Surprised       0.31      0.31      0.31        32
    accuracy                           0.39       240
   macro avg       0.38      0.38      0.36       240
weighted avg       0.38      0.39      0.37       240
```

---

## 7. ASR Evaluation & JiWER Benchmarks

### 7.1 Issue Identified During Audit
In the original `evaluate_asr.py`, a static list of four strings was evaluated against identical strings, artificially reporting a 0.00% WER without executing actual audio through the models.

### 7.2 Remediation Implemented
`evaluate_asr.py` was refactored to:
1. Dynamically select a stratified multi-speaker subset of genuine RAVDESS audio recordings across actors.
2. Extract the true ground truth reference statement (`"Kids are talking by the door"` or `"Dogs are sitting by the door"`) using the verified RAVDESS statement code parser.
3. Feed the audio through both `openai/whisper-base` and `facebook/wav2vec2-base-960h`.
4. Calculate standard Word Error Rate (WER) and Character Error Rate (CER) via JiWER with standard punctuation normalization.

### 7.3 Measured ASR Results
Evaluated across 16 multi-speaker recordings:

| Model Architecture | Word Error Rate (WER) | Character Error Rate (CER) | Word-Level Accuracy |
| :--- | :---: | :---: | :---: |
| **OpenAI Whisper Base (Primary)** | **1.04%** | **0.89%** | **98.96%** |
| **Facebook Wav2Vec 2.0 Base (Baseline)** | **2.08%** | **0.45%** | **97.92%** |

- **Sample Analysis:**
  - Whisper accurately transcribed 15 out of 16 sentences verbatim, with a single minor insertion on an actor's initial vocal hesitation (*"Hmm, kids are talking by the door"*).
  - Wav2Vec 2.0 achieved an ultra-low CER of 0.45%, exhibiting minor phonetic confusions on unvoiced plosives (e.g., *"Kits"* for *"Kids"*).
- **Scope & Generalization Limitation:** The reported ~1% to 2% WER is representative of clean, studio-recorded read speech on fixed statements. Spontaneous conversational speech with background noise will exhibit higher error rates.

---

## 8. Streamlit Dashboard Audit

The user interface was audited against the 15 functional requirements:

| # | Requirement | Implementation Verification |
| :-: | :--- | :--- |
| 1 | Audio upload | Functional via `st.file_uploader(type=['wav'])` with format validation |
| 2 | Audio playback | Functional via native HTML5 audio stream (`st.audio`) |
| 3 | Audio metadata | Displays filename, duration (s), sampling rate (Hz), and channel format |
| 4 | ASR model selection | Dropdown toggle: `openai/whisper-base` vs. `facebook/wav2vec2-base-960h` |
| 5 | SER model selection | Dropdown toggle: `Gradient Boosting (Latest)` vs. `Random Forest (Baseline)` |
| 6 | Simultaneous execution | Single action button triggers both ASR and SER sequentially with progress bar |
| 7 | Actual speech transcription | Renders real model transcript in clean blockquote container |
| 8 | Predicted emotion | High-impact uppercase typography (`HAPPY`, `NEUTRAL`, `ANGRY`) with estimated probability |
| 9 | 8-class probability chart | Minimalist Plotly horizontal bar chart highlighting winning class in deep slate |
| 10 | Waveform visualization | Downsampled time-domain waveform using Plotly line scatter |
| 11 | Mel-spectrogram heatmap | 64-band mel-frequency power spectrogram computed via `librosa` and Plotly `imshow` |
| 12 | Pitch ($F_0$) visualization | Fundamental frequency trajectory traced via `librosa.yin` |
| 13 | RMS energy visualization | RMS amplitude contour overlaid with dotted trajectory |
| 14 | Evaluation metrics | Clean metric boxes for Accuracy, Precision, Recall, Macro F1, WER, CER |
| 15 | Confusion matrix | Full $8 \times 8$ actor-independent test confusion matrix visualization |

### Modern Minimal Theme Compliance
- **Typography:** Inter font family applied globally via CSS.
- **Palette:** Slate `#0F172A`, Muted Zinc `#64748B`, Subtle Borders `#E2E8F0`, Clean Background `#F8FAFC`.
- **Zero Emojis:** Confirmed 0 emojis in code, headers, buttons, labels, and chart outputs.

---

## 9. Automated Testing & SQA Suite

The automated test suite in `tests/` was expanded from 19 to **22 comprehensive unit and integration tests**:

```powershell
python -m pytest tests/ -v
================== 22 passed, 1 warning in 72.63s (0:01:12) ==================
```

### Breakdown of Test Cases
1. `TestRAVDESSParsing::test_valid_speech_filename` – PASSED
2. `TestRAVDESSParsing::test_male_actor_odd_id` – PASSED
3. `TestRAVDESSParsing::test_all_eight_emotions_mapped` – PASSED
4. `TestRAVDESSParsing::test_song_modality_detection` – PASSED
5. `TestRAVDESSParsing::test_invalid_filename_length` – PASSED
6. `TestRAVDESSParsing::test_invalid_emotion_code` – PASSED
7. `TestRAVDESSParsing::test_invalid_actor_code` – PASSED
8. `TestAudioFeatureExtraction::test_feature_dimension_and_names` – PASSED
9. `TestAudioFeatureExtraction::test_pitch_detected_on_harmonic_tone` – PASSED
10. `TestAudioFeatureExtraction::test_silent_audio_safety` – PASSED
11. `TestAudioFeatureExtraction::test_short_audio_handling` – PASSED
12. `TestAudioFeatureExtraction::test_stereo_to_mono_and_resampling` – PASSED
13. `TestDualPipelineInference::test_emotion_model_loading` – PASSED
14. `TestDualPipelineInference::test_gradient_boosting_emotion_prediction` – PASSED
15. `TestDualPipelineInference::test_random_forest_emotion_prediction` – PASSED
16. `TestDualPipelineInference::test_whisper_transcription` – PASSED
17. `TestDualPipelineInference::test_wav2vec2_transcription` – PASSED
18. `TestDualPipelineInference::test_dual_pipeline_same_audio` – PASSED
19. `TestDualPipelineInference::test_silence_guard` – PASSED
20. `TestDualPipelineInference::test_actor_independent_split_integrity` – PASSED
21. `TestDualPipelineInference::test_asr_evaluation_metrics_function` – PASSED
22. `TestStreamlitAppIntegrity::test_app_compilation` – PASSED

---

## 10. Hardware & Latency Benchmark

Empirical latencies were profiled with strict methodological separation between **Cold Model Loading** (disk-to-memory initialization) and **Warm Pure Inference** on the host environment (Intel CPU, Windows 11, PyTorch CPU mode):

### 10.1 Cold Model Loading (One-Time Startup Overhead)
| Component | Measured Startup Latency | Initialization Details |
| :--- | :---: | :--- |
| **SER Model Artifact (`models/emotion_model.pkl`)** | ~38.85 s | Deserialization of scaler, HGB, and RF ensembles |
| **Facebook Wav2Vec 2.0 Base Engine** | ~1.08 s | PyTorch 95M-parameter weight loading & CTC processor |
| **OpenAI Whisper Base Engine** | ~123.05 s | PyTorch 74M-parameter weights & autoregressive decoder setup |

### 10.2 Warm Audio Processing & Pure Inference (Per Sample)
| Pipeline Stage | Measured Latency | Subroutine Context |
| :--- | :---: | :--- |
| **Audio Loading & Resampling (48 kHz to 16 kHz)** | ~4.31 s | `soundfile` buffer read, peak-normalization, librosa resample |
| **Acoustic Feature Extraction (61 Features)** | ~186.9 ms | Librosa YIN pitch tracking, MFCCs, RMS, spectral centroid |
| **Pure SER Inference (HistGradientBoosting)** | **~130.8 ms** | Scikit-learn histogram decision tree prediction & probabilities |
| **Pure SER Inference (Random Forest)** | **~87.9 ms** | Scikit-learn 150-tree ensemble prediction & probabilities |
| **Pure ASR Inference (Facebook Wav2Vec 2.0 Base)** | **~342.0 ms (0.34 s)** | Non-autoregressive CTC acoustic decoding |
| **Pure ASR Inference (OpenAI Whisper Base)** | **~1.34 s** | Autoregressive beam search decoding (`max_new_tokens=64`) |

### 10.3 End-to-End Warm Single-Trigger Dual Inference
| Full Pipeline Combination | Total Warm Turnaround | Execution Flow |
| :--- | :---: | :--- |
| **Wav2Vec 2.0 Base + HistGradientBoosting** | **~4.97 s** | Audio IO + Resampling + ASR CTC + 61 Feats + SER |
| **OpenAI Whisper Base + HistGradientBoosting** | **~5.96 s** | Audio IO + Resampling + ASR Whisper + 61 Feats + SER |

> [!NOTE]
> **Single-Trigger Dual Sequential Processing:** The system architecture executes both pipelines from a single user action. The pipelines run sequentially in the application thread (ASR transcription followed by SER feature extraction and classification) rather than via multithreaded concurrency.

---

## 11. Issues Identified and Remediations

| Issue Identified | Severity | Remediation Applied |
| :--- | :---: | :--- |
| **1. Mock ASR Benchmark:** `evaluate_asr.py` was comparing hardcoded text strings rather than real audio. | High | Completely refactored `evaluate_asr.py` to run genuine audio inference on stratified RAVDESS recordings with JiWER. |
| **2. Synthetic Demo Tones Defaulting to Calm:** Demo folder previously had artificial sine clips lacking human vocal formants. | High | Replaced with 8 authentic human speech recordings from RAVDESS covering all 8 emotions. |
| **3. Plotly `titlefont` Deprecation Error:** Plotly threw a `ValueError` for `titlefont`. | Medium | Replaced all instances of `titlefont` with modern `title_font` in `app.py`. |
| **4. Incomplete Test Coverage:** Split integrity and ASR evaluation functions lacked test assertions. | Medium | Added 3 new unit tests to `test_inference.py` (totaling 22 passed tests). |
| **5. Numba/Loky Core Detection Warning on Windows:** Subprocess warning on core detection. | Low | Added `LOKY_MAX_CPU_COUNT=4` and explicit local `.cache/` directory overrides. |
| **6. Legacy Emojis in Core Backend:** `emotion_detection.py` contained residual emojis in `EMOTION_ICONS`. | Low | Purged all emoji mappings in `emotion_detection.py` to guarantee zero non-ASCII characters across the entire Python backend. |
| **7. Ambiguous Button Terminology:** Button in `app.py` previously read "Run Simultaneous Inference". | Low | Updated button to "Run Dual Pipeline Analysis" to accurately describe single-trigger dual sequential inference. |

---

## 12. Demonstration Readiness & Faculty Presentation Guide

The project is **100% verified, technically defensible, and ready for academic demonstration**:

### Demonstration Checklist
1. **Launch Dashboard:**
   ```powershell
   streamlit run app.py
   ```
2. **Demonstrate Speech Emotion Detection:**
   - Select `Demo 03: Happy` $\rightarrow$ Classifies to **Happy** with high estimated probability.
   - Select `Demo 05: Angry` $\rightarrow$ Classifies to **Angry** with elevated pitch and RMS energy.
   - Select `Demo 01: Neutral` $\rightarrow$ Classifies to **Neutral**.
3. **Demonstrate Speech-to-Text:**
   - Shows real-time transcription: *"Kids are talking by the door."*
   - Toggle between **Whisper Base** and **Wav2Vec 2.0** to explain autoregressive vs. CTC decoding differences.
4. **Demonstrate Signal Visualizations:**
   - Show how the pitch tracking contour ($F_0$) spikes during *Angry* vs. flattens during *Calm* or *Sad*.
5. **Demonstrate Methodological Rigor:**
   - Present the Actor-Independent split: Actors 21–24 were never seen during training.
   - Show the 43.75% 8-class accuracy (3.5x random baseline) and the confusion matrix in the dashboard.

