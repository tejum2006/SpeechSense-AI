<p align="center">
  <img src="assets/logo.png" width="140" alt="SpeechSense AI Logo" />
</p>

# SpeechSense AI: Dual-Pipeline Speech Recognition and Vocal Emotion Analysis System

> **"Understanding speech beyond words."**
> A dual-pipeline academic and research dashboard integrating Automatic Speech Recognition (**OpenAI Whisper** and **Facebook Wav2Vec 2.0**) and Speech Emotion Recognition (**Histogram Gradient Boosting** and **Random Forest** on RAVDESS).

---

## 1. Project Introduction

**SpeechSense AI** is a dual-pipeline speech analysis system designed for academic demonstration and vocal acoustic research. It processes speech audio recordings via single-trigger dual sequential inference through two independent pipelines:

1. **Automatic Speech Recognition (ASR):** Converts spoken acoustic waveforms into natural English text using:
   - **OpenAI Whisper Base (Primary ASR Model):** Sequence-to-sequence transformer with robust punctuation and token decoding.
   - **Facebook Wav2Vec 2.0 Base (Academic Baseline):** Connectionist Temporal Classification (CTC) acoustic model.
2. **Speech Emotion Recognition (SER):** Predicts emotional states across 8 categories using:
   - **Histogram Gradient Boosting (Primary Model):** Non-linear gradient boosted decision trees.
   - **Random Forest Classifier (Academic Baseline):** Ensemble bagging trees.
   - Evaluated under strict **actor-independent validation** on the RAVDESS speech dataset.

Both pipelines evaluate the exact same uploaded recording, delivering transcripts, emotion classifications, estimated probability distributions, and acoustic signal analytics.

---

## 2. Supported Emotion Categories

1. **Neutral**
2. **Calm**
3. **Happy**
4. **Sad**
5. **Angry**
6. **Fearful**
7. **Disgust**
8. **Surprised**

---

## 3. Technology Stack

| Category | Component | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core runtime environment |
| **ASR Models** | `openai/whisper-base` & `facebook/wav2vec2-base-960h` | State-of-the-art transformer speech-to-text |
| **SER Models** | `HistGradientBoostingClassifier` & `RandomForestClassifier` | Supervised acoustic emotion classification |
| **Deep Learning** | PyTorch (`torch`, `torchaudio`), Transformers | Neural tensor execution engine |
| **Audio Processing** | Librosa, SoundFile, SciPy, NumPy | Signal processing, resampling, 61 acoustic features |
| **Frontend** | Streamlit | Responsive web dashboard |
| **Visualizations** | Plotly, Matplotlib | Interactive waveforms, spectrogram heatmaps, F0 contours |
| **Evaluation** | JiWER, Scikit-learn, Pytest | WER/CER metrics & automated unit tests |

---

## 4. Acoustic Feature Representation (61 Dimensions)

The SER pipeline extracts a fixed 61-dimensional acoustic feature vector:
- **MFCCs (13 coefficients):** Mean & standard deviation across time (**26 features**)
- **Delta MFCCs (13 coefficients):** First-order temporal derivatives (**26 features**)
- **Fundamental Frequency $F_0$ (Pitch):** Tracking via `librosa.yin` (Mean, Std, Min, Max) (**4 features**)
- **RMS Signal Energy:** Signal intensity (Mean, Std, Max) (**3 features**)
- **Spectral Centroid:** Spectral brightness descriptor (Mean, Std) (**2 features**)

---

## 5. Quick Start Guide (Windows)

### Step 1: Clone or Navigate to Directory
```powershell
cd "c:\Users\DELL\OneDrive\Desktop\Speech Emotion"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Launch Web Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 6. Model Training & Evaluation

### Train Emotion Models (Dual SER Pipeline)
To train both Gradient Boosting and Random Forest models on RAVDESS with actor-independent splitting (Actors 1–20 for training, Actors 21–24 for testing):
```powershell
python train_emotion_model.py
```

### Inspect Evaluation Metrics & Confusion Matrix
```powershell
python evaluate_emotion_model.py
```

### Benchmark ASR Word Error Rate (WER)
```powershell
python evaluate_asr.py
```

### Run Full Automated Test Suite
```powershell
python -m pytest tests/ -v
```

---

## 7. Project Structure

```
Speech Emotion/
│
├── app.py                      # Main Streamlit dashboard application
├── train_emotion_model.py      # Actor-independent model training pipeline
├── speech_recognition.py       # Dual ASR inference module (Whisper & Wav2Vec 2.0)
├── emotion_detection.py        # Dual SER inference module (Gradient Boosting & Random Forest)
├── feature_extraction.py       # 61 acoustic features extraction module
├── evaluate_emotion_model.py   # SER evaluation & confusion matrix generator
├── evaluate_asr.py             # ASR Word Error Rate (WER/CER) evaluation script
│
├── models/
│   ├── emotion_model.pkl       # Bundled scaler + trained SER models
│   ├── evaluation_report.json  # SER evaluation metrics
│   └── asr_evaluation_report.json # ASR WER benchmark metrics
│
├── dataset/
│   ├── RAVDESS/                # Extracted RAVDESS dataset audio files
│   └── create_demo_dataset.py  # Demonstration dataset generator
│
├── sample_audio/               # 5 preloaded demonstration WAV files
│   └── generate_demo_samples.py
│
├── assets/
│   └── confusion_matrix.png    # 300 DPI publishable confusion matrix plot
│
├── tests/
│   ├── test_feature_extraction.py # 12 unit tests
│   └── test_inference.py          # 7 integration & dual-model tests
│
├── .streamlit/
│   └── config.toml             # Localhost WebSocket & server configuration
│
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation
└── .gitignore                  # Git ignore rules
```

---

## 8. References

1. Radford, A., et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*. International Conference on Machine Learning (ICML).
2. Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS.
3. Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. PLoS ONE.
