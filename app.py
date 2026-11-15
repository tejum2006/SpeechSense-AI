"""
SpeechSense AI - Premium Dark Mode Web Dashboard
================================================
Dual-pipeline Automatic Speech Recognition (OpenAI Whisper Base / Wav2Vec 2.0)
and Speech Emotion Recognition (Histogram Gradient Boosting / Random Forest).

Theme: Premium Dark Glassmorphism, Ambient Gradient Accents, Modern Geometric Typography.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Project paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PNG_PATH = ASSETS_DIR / "logo.png"
LOGO_SVG_PATH = ASSETS_DIR / "logo.svg"
MODELS_DIR = BASE_DIR / "models"
EMOTION_MODEL_FILE = MODELS_DIR / "emotion_model.pkl"
EVAL_REPORT_FILE = MODELS_DIR / "evaluation_report.json"
ASR_REPORT_FILE = MODELS_DIR / "asr_evaluation_report.json"
SAMPLE_AUDIO_DIR = BASE_DIR / "sample_audio"

# Configure page settings
st.set_page_config(
    page_title="SpeechSense AI | Dual-Pipeline Speech Intelligence",
    page_icon=str(LOGO_PNG_PATH) if LOGO_PNG_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load base64 encoded logo for standalone embedding
LOGO_BASE64 = ""
if LOGO_PNG_PATH.exists():
    with open(LOGO_PNG_PATH, "rb") as f:
        LOGO_BASE64 = base64.b64encode(f.read()).decode("utf-8")

# Standard 8-Class Emotion Labels
EMOTION_LABELS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
]

# Vibrant Emotion Color Palette with Glowing HEX definitions
EMOTION_COLORS = {
    "Neutral": "#94A3B8",    # Slate Silver
    "Calm": "#06B6D4",       # Electric Cyan
    "Happy": "#FBBF24",      # Radiant Sun Gold
    "Sad": "#3B82F6",        # Royal Cobalt Blue
    "Angry": "#EF4444",      # Vivid Fiery Crimson
    "Fearful": "#F97316",    # Electric Tangerine
    "Disgust": "#10B981",    # Neon Emerald Green
    "Surprised": "#A855F7",  # Electric Violet Orchid
}

EMOTION_DISCLAIMER = (
    "Acoustic classification scores estimate probabilistic vocal characteristics "
    "and do not constitute a clinical determination of genuine psychological state."
)

# ---------------------------------------------------------
# THEME: COMPLETE PREMIUM DARK MODE & MODERN GEOMETRIC FONTS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #F1F5F9 !important;
        background-color: #080B11 !important;
    }
    
    /* Obsidian Canvas with Ambient Radial Mesh */
    .stApp {
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
            radial-gradient(circle at 90% 20%, rgba(236, 72, 153, 0.06) 0%, transparent 40%),
            radial-gradient(circle at 10% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 40%) !important;
        background-attachment: fixed !important;
    }
    
    /* Universal text colors for high readability */
    p, span, label, div, h1, h2, h3, h4, h5, h6, th, td {
        color: #F1F5F9;
    }

    /* Global Layout Constraints */
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1240px;
    }

    /* Primary Header Container */
    .app-header {
        background: linear-gradient(180deg, rgba(19, 24, 38, 0.85) 0%, rgba(12, 16, 26, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.4rem 1.85rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 30%, #EC4899 65%, #F59E0B 100%);
    }
    .header-brand {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.85rem;
    }
    .brand-logo-img {
        width: 52px;
        height: 52px;
        border-radius: 13px;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.45), 0 0 35px rgba(6, 182, 212, 0.2);
        flex-shrink: 0;
        display: block;
        transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out;
    }
    .brand-logo-img:hover {
        transform: scale(1.05);
        box-shadow: 0 0 28px rgba(139, 92, 246, 0.65), 0 0 45px rgba(6, 182, 212, 0.35);
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 45%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }
    .header-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
        letter-spacing: 0.01em;
        margin-top: 0.3rem;
    }
    .status-group {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        flex-wrap: wrap;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.28rem 0.7rem;
        border-radius: 6px;
        backdrop-filter: blur(8px);
    }
    .status-dot-emerald {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
        display: inline-block;
        margin-right: 6px;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.7);
    }
    .status-dot-amber {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #F59E0B;
        display: inline-block;
        margin-right: 6px;
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.7);
    }

    /* Modern Dark Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 2px;
        margin-bottom: 1.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 6px 18px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #94A3B8;
        border: 1px solid transparent;
        border-bottom: none;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background-color: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-bottom: 2px solid #818CF8 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 -4px 12px rgba(99, 102, 241, 0.2);
    }

    /* Premium Dark Card Containers with Glassmorphism */
    .card-shell {
        background: linear-gradient(180deg, rgba(18, 24, 38, 0.85) 0%, rgba(12, 16, 26, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.35rem 1.6rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 1.15rem;
        backdrop-filter: blur(12px);
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.7rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #A5B4FC;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0;
    }
    .card-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #94A3B8;
        font-weight: 500;
    }

    /* Audio Deck Container */
    .deck-container {
        background: linear-gradient(180deg, rgba(18, 24, 38, 0.85) 0%, rgba(12, 16, 26, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
    }

    /* Streamlit Radio Buttons - Fully Visible & Interactive */
    div[data-testid="stRadio"] {
        background: rgba(13, 17, 28, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
    }
    div[data-testid="stRadio"] label {
        color: #F1F5F9 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label p {
        color: #F1F5F9 !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 1.8rem;
    }

    /* Streamlit Selectboxes */
    div[data-testid="stSelectbox"] label p {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    div[data-baseweb="select"] {
        background-color: #111625 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #111625 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }
    li[role="option"] {
        background-color: #111625 !important;
        color: #F1F5F9 !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }

    /* Streamlit File Uploader - Fully Visible, Glowing & Clean */
    div[data-testid="stFileUploader"] {
        background: rgba(13, 17, 28, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.9rem;
    }
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] label p {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(17, 22, 37, 0.8) !important;
        border: 1.5px dashed rgba(139, 92, 246, 0.5) !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        text-align: center;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #A78BFA !important;
        background: rgba(25, 33, 56, 0.9) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.25) !important;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #E2E8F0 !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1.15rem !important;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3) !important;
    }
    div[data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%) !important;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.5) !important;
    }
    div[data-testid="stFileUploaderFileData"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.8rem !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stFileUploaderFileData"] * {
        color: #F8FAFC !important;
    }

    /* Glowing Monospace Telemetry Badges */
    .meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }
    .tag-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
    }
    .tag-blue {
        background: rgba(59, 130, 246, 0.12);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
    }
    .tag-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
    }
    .tag-purple {
        background: rgba(139, 92, 246, 0.12);
        color: #C4B5FD;
        border: 1px solid rgba(139, 92, 246, 0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
    }
    .tag-cyan {
        background: rgba(6, 182, 212, 0.12);
        color: #67E8F9;
        border: 1px solid rgba(6, 182, 212, 0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
    }
    .tag-slate {
        background: rgba(148, 163, 184, 0.1);
        color: #CBD5E1;
        border: 1px solid rgba(148, 163, 184, 0.25);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
    }

    /* Dual Stream Intelligence Displays */
    .transcript-stream {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.5) 100%);
        border-left: 4px solid #6366F1;
        border-radius: 6px;
        padding: 1.25rem 1.45rem;
        font-size: 1.2rem;
        font-weight: 500;
        color: #F8FAFC;
        line-height: 1.6;
        min-height: 80px;
        display: flex;
        align-items: center;
        margin-bottom: 0.95rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .emotion-badge-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem 1.35rem;
        margin-bottom: 0.95rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .emotion-title-text {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        text-transform: uppercase;
        line-height: 1.1;
    }
    .prob-indicator-pill {
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.42rem 0.85rem;
        border-radius: 6px;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Stat Metric Tiles with Gradient Top Borders */
    .stat-tile {
        background: linear-gradient(180deg, rgba(18, 24, 38, 0.9) 0%, rgba(12, 16, 26, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.25rem 0.95rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
    }
    .stat-tile-emerald { border-top: 4px solid #10B981; }
    .stat-tile-blue { border-top: 4px solid #3B82F6; }
    .stat-tile-purple { border-top: 4px solid #8B5CF6; }
    .stat-tile-amber { border-top: 4px solid #F59E0B; }

    .stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .stat-lbl {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.45rem;
    }

    /* Primary Action Button (Gradient Purple / Indigo) */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #9333EA 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1.45rem !important;
        box-shadow: 0 4px 20px -2px rgba(124, 58, 237, 0.5) !important;
        transition: all 0.2s ease-in-out !important;
        letter-spacing: 0.01em !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 50%, #7E22CE 100%) !important;
        border-color: rgba(255, 255, 255, 0.35) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 25px -2px rgba(124, 58, 237, 0.75) !important;
        transform: translateY(-1px) !important;
    }

    /* Sleek Dark Audio Player */
    div[data-testid="stAudio"] {
        margin-top: 0.5rem;
    }
    audio {
        width: 100%;
        border-radius: 8px;
        background: #111625;
    }

    /* Sidebar Dark Styling */
    section[data-testid="stSidebar"] {
        background-color: #0A0D15 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F1F5F9;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
    }

    /* Captions and Subtext */
    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
    }

    /* Info / Warning / Alert Boxes */
    div[data-testid="stAlert"] {
        background: rgba(18, 24, 38, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
    }
    div[data-testid="stAlert"] * {
        color: #F1F5F9 !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CACHED MODEL LOADERS
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_asr_engine(model_key: str):
    from speech_recognition import get_speech_recognizer
    return get_speech_recognizer(model_type=model_key)


@st.cache_resource(show_spinner=False)
def load_ser_engine():
    from emotion_detection import get_emotion_detector
    det = get_emotion_detector(model_path=EMOTION_MODEL_FILE)
    return det if det.is_loaded else None


def check_models_status():
    ser_ok = EMOTION_MODEL_FILE.exists()
    asr_ok = True
    return asr_ok, ser_ok


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    if LOGO_BASE64:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.15rem; padding-bottom: 0.85rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                <img src="data:image/png;base64,{LOGO_BASE64}" style="width: 36px; height: 36px; border-radius: 9px; box-shadow: 0 0 12px rgba(139, 92, 246, 0.4);" alt="Logo" />
                <div>
                    <div style="font-weight: 800; font-size: 1.05rem; letter-spacing: -0.03em; color: #FFFFFF; line-height: 1.15;">SpeechSense AI</div>
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 500;">Acoustic Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="card-title" style="margin-bottom: 0.8rem; color: #A5B4FC;">Pipeline Configuration</div>', unsafe_allow_html=True)
    
    st.caption("Active Machine Learning Models")
    asr_choice = st.selectbox(
        "Speech Recognition (ASR):",
        [
            "OpenAI Whisper Base (Primary ASR Model)",
            "Facebook Wav2Vec 2.0 (Baseline)",
        ],
        index=0,
        help="Select between OpenAI Whisper Base or Facebook Wav2Vec 2.0.",
    )
    selected_asr_model = "openai/whisper-base" if "Whisper" in asr_choice else "facebook/wav2vec2-base-960h"

    ser_detector = load_ser_engine()
    available_ser_models = (
        ser_detector.get_available_models()
        if ser_detector
        else ["Gradient Boosting (Latest)", "Random Forest (Baseline)"]
    )
    ser_choice = st.selectbox(
        "Emotion Recognition (SER):",
        available_ser_models,
        index=0,
        help="Select between Modern Histogram Gradient Boosting or Random Forest.",
    )

    st.markdown("---")
    st.caption("System Specifications")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.6;">
            &bull; <strong style="color: #F1F5F9;">Audio Target:</strong> 16,000 Hz Mono PCM<br>
            &bull; <strong style="color: #F1F5F9;">Feature Vector:</strong> 61 Acoustic Descriptors<br>
            &bull; <strong style="color: #F1F5F9;">Corpus:</strong> RAVDESS (1,440 recordings)<br>
            &bull; <strong style="color: #F1F5F9;">Validation:</strong> Actor-Independent (Zero Leakage)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("Reset Analysis State", use_container_width=True):
        for k in ["asr_result", "ser_result", "analyzed_wave", "analyzed_sr", "analyzed_file"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ---------------------------------------------------------
# TOP APPLICATION HEADER
# ---------------------------------------------------------
asr_ready, ser_ready = check_models_status()
asr_badge = "Whisper Base" if "Whisper" in asr_choice else "Wav2Vec 2.0"
ser_badge = "Gradient Boosting" if "Gradient" in ser_choice else "Random Forest"

logo_html = f'<img src="data:image/png;base64,{LOGO_BASE64}" class="brand-logo-img" alt="SpeechSense AI Logo" />' if LOGO_BASE64 else ""

st.markdown(
    f"""
    <div class="app-header">
        <div class="header-brand">
            <div style="display: flex; align-items: center; gap: 1.15rem;">
                {logo_html}
                <div>
                    <h1 class="header-title">SpeechSense AI</h1>
                    <div class="header-subtitle">Dual-Pipeline Speech Recognition and Vocal Emotion Analysis System</div>
                </div>
            </div>
            <div class="status-group">
                <span class="status-pill"><span class="status-dot-emerald"></span>ASR: {asr_badge}</span>
                <span class="status-pill"><span class="status-dot-emerald" style="background: {'#10B981' if ser_ready else '#F59E0B'};"></span>SER: {ser_badge}</span>
                <span class="status-pill">16 kHz Mono</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ser_ready:
    st.warning("Emotion model artifact not found. Execute 'python train_emotion_model.py' to generate.")


# ---------------------------------------------------------
# TOP LEVEL VIEW NAVIGATION (STUDIO / EVALUATION / ARCHITECTURE)
# ---------------------------------------------------------
view_studio, view_eval, view_arch = st.tabs([
    "Analysis Studio",
    "Performance & Benchmarks",
    "Architecture & 61 Features",
])


# =========================================================
# TAB 1: ANALYSIS STUDIO
# =========================================================
with view_studio:
    # 1. AUDIO DECK & SOURCE SELECTION
    st.markdown(
        """
        <div class="deck-container">
            <div class="card-header">
                <span class="card-title">Audio Input Deck</span>
                <span class="card-meta">Standard Uncompressed PCM (16 kHz / 48 kHz Mono or Stereo)</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size: 0.82rem; font-weight: 700; color: #A5B4FC; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.45rem;">1. Select Audio Source</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Choose audio source:",
        [
            "Demonstration Library (8 RAVDESS Emotions)",
            "Upload Custom Audio File (WAV)",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    audio_bytes = None
    audio_filename = ""

    if input_mode == "Demonstration Library (8 RAVDESS Emotions)":
        st.markdown(
            '<div style="font-size: 0.82rem; font-weight: 700; color: #A5B4FC; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.85rem; margin-bottom: 0.35rem;">2. Choose Human Speech Recording</div>',
            unsafe_allow_html=True,
        )
        demo_files = sorted(SAMPLE_AUDIO_DIR.glob("*.wav")) if SAMPLE_AUDIO_DIR.exists() else []
        demo_labels = {
            "demo_01_neutral.wav": "Demo 01: Neutral (Actor Speech)",
            "demo_02_calm.wav": "Demo 02: Calm (Actor Speech)",
            "demo_03_happy.wav": "Demo 03: Happy (Actor Speech)",
            "demo_04_sad.wav": "Demo 04: Sad (Actor Speech)",
            "demo_05_angry.wav": "Demo 05: Angry (Actor Speech)",
            "demo_06_fearful.wav": "Demo 06: Fearful (Actor Speech)",
            "demo_07_disgust.wav": "Demo 07: Disgust (Actor Speech)",
            "demo_08_surprised.wav": "Demo 08: Surprised (Actor Speech)",
        }
        if demo_files:
            selected_demo = st.selectbox(
                "Choose an authentic human speech sample from the library:",
                demo_files,
                format_func=lambda f: demo_labels.get(f.name, f.name),
                label_visibility="collapsed",
            )
            if selected_demo:
                with open(selected_demo, "rb") as fp:
                    audio_bytes = fp.read()
                audio_filename = selected_demo.name
        else:
            st.info("No sample files located in sample_audio/.")
    else:
        st.markdown(
            '<div style="font-size: 0.82rem; font-weight: 700; color: #A5B4FC; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.85rem; margin-bottom: 0.35rem;">2. Upload Audio File (WAV Format)</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload a speech audio file (WAV):",
            type=["wav"],
            help="Accepts 16-bit or 24-bit PCM WAV audio recordings.",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            audio_filename = uploaded_file.name

    if audio_bytes is not None:
        from feature_extraction import load_audio, TARGET_SAMPLE_RATE
        try:
            y, orig_sr = load_audio(io.BytesIO(audio_bytes), target_sr=TARGET_SAMPLE_RATE)
            duration_sec = len(y) / float(TARGET_SAMPLE_RATE)

            c_play, c_action = st.columns([3, 2])
            with c_play:
                st.audio(audio_bytes, format="audio/wav")
                st.markdown(
                    f"""
                    <div class="meta-bar">
                        <span class="tag-amber">File: {audio_filename}</span>
                        <span class="tag-blue">Duration: {duration_sec:.2f}s</span>
                        <span class="tag-emerald">Sampling: {TARGET_SAMPLE_RATE:,} Hz</span>
                        <span class="tag-purple">Format: PCM Mono</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_action:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                run_btn = st.button("Run Dual Pipeline Analysis", use_container_width=True)
        except Exception as err:
            st.error(f"Error loading audio: {err}")
            run_btn = False
    else:
        run_btn = False

    st.markdown("</div>", unsafe_allow_html=True)

    # 2. INFERENCE EXECUTION PIPELINE
    if run_btn and audio_bytes is not None:
        progress_bar = st.progress(0, text="Initializing inference engines...")

        # Step 1: ASR Pipeline
        progress_bar.progress(35, text=f"Transcribing linguistic text ({asr_badge})...")
        recognizer = load_asr_engine(selected_asr_model)
        if recognizer:
            asr_res = recognizer.transcribe(io.BytesIO(audio_bytes))
        else:
            asr_res = {"status": "error", "transcript": "(ASR model unavailable)", "inference_time_sec": 0.0}

        # Step 2: SER Pipeline
        progress_bar.progress(75, text=f"Extracting 61 acoustic descriptors ({ser_badge})...")
        detector = load_ser_engine()
        if detector:
            ser_res = detector.predict(io.BytesIO(audio_bytes), model_name=ser_choice)
        else:
            ser_res = {
                "status": "not_trained",
                "predicted_emotion": "Unavailable",
                "confidence_score": 0.0,
                "probabilities": {e: 0.0 for e in EMOTION_LABELS},
            }

        progress_bar.progress(100, text="Inference complete.")
        time.sleep(0.15)
        progress_bar.empty()

        st.session_state["asr_result"] = asr_res
        st.session_state["ser_result"] = ser_res
        st.session_state["analyzed_wave"] = y
        st.session_state["analyzed_sr"] = TARGET_SAMPLE_RATE
        st.session_state["analyzed_file"] = audio_filename

    # 3. DISPLAY RESULTS IF ANALYZED
    if "asr_result" in st.session_state and "ser_result" in st.session_state:
        asr_data = st.session_state["asr_result"]
        ser_data = st.session_state["ser_result"]
        y_wave = st.session_state["analyzed_wave"]
        sr_val = st.session_state["analyzed_sr"]

        col_asr, col_ser = st.columns(2)

        # Panel 1: ASR Linguistic Output
        with col_asr:
            st.markdown(
                f"""
                <div class="card-shell" style="border-top: 3px solid #6366F1;">
                    <div class="card-header">
                        <span class="card-title">Speech-to-Text Transcription</span>
                        <span class="card-meta">Model: {asr_data.get('model_name', selected_asr_model)}</span>
                    </div>
                    <div class="transcript-stream">
                        "{asr_data.get('transcript', '')}"
                    </div>
                    <div class="meta-bar">
                        <span class="tag-blue">Latency: {asr_data.get('inference_time_sec', 0.0):.2f}s</span>
                        <span class="tag-emerald">Device: {asr_data.get('device', 'cpu').upper()}</span>
                        <span class="tag-amber">Words: {len(asr_data.get('transcript', '').split())}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Panel 2: SER Vocal Emotion Output
        with col_ser:
            pred_emo = ser_data.get("predicted_emotion", "Neutral")
            pred_prob = ser_data.get("confidence_score", 0.0) * 100
            clf_name = ser_data.get("model_name", ser_choice)
            dominant_color = EMOTION_COLORS.get(pred_emo, "#8B5CF6")

            st.markdown(
                f"""
                <div class="card-shell" style="border-top: 3px solid {dominant_color};">
                    <div class="card-header">
                        <span class="card-title">Vocal Emotion Classification</span>
                        <span class="card-meta">Model: {clf_name}</span>
                    </div>
                    <div class="emotion-badge-container">
                        <div>
                            <span style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">Dominant Vocal Emotion</span>
                            <div class="emotion-title-text" style="color: {dominant_color}; text-shadow: 0 0 20px {dominant_color}40;">{pred_emo}</div>
                        </div>
                        <div class="prob-indicator-pill" style="background: {dominant_color};">{pred_prob:.1f}% Estimated Prob</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Colorful Vibrant Horizontal Probability Bar Chart on Dark Canvas
            probs = ser_data.get("probabilities", {})
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=False)
            labels = [it[0] for it in sorted_items]
            vals = [it[1] * 100 for it in sorted_items]
            bar_colors = [EMOTION_COLORS.get(l, "#8B5CF6") for l in labels]

            fig_p = go.Figure(
                go.Bar(
                    x=vals,
                    y=labels,
                    orientation="h",
                    marker=dict(
                        color=bar_colors,
                        line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
                    ),
                    text=[f"{v:.1f}%" for v in vals],
                    textposition="outside",
                    textfont=dict(size=9.5, family="JetBrains Mono, monospace", color="#E2E8F0"),
                )
            )
            fig_p.update_layout(
                xaxis=dict(
                    range=[0, 115],
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.06)",
                    zeroline=False,
                    tickfont=dict(size=9, color="#94A3B8", family="JetBrains Mono, monospace"),
                ),
                yaxis=dict(
                    tickfont=dict(size=10, color="#F8FAFC", family="Plus Jakarta Sans, sans-serif"),
                ),
                height=215,
                margin=dict(l=10, r=25, t=5, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_p, use_container_width=True)
            st.caption(f"{EMOTION_DISCLAIMER}")
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. ACOUSTIC SIGNAL STUDIO
        st.markdown(
            """
            <div class="card-shell" style="border-top: 3px solid #3B82F6;">
                <div class="card-header">
                    <span class="card-title">Acoustic Signal Studio</span>
                    <span class="card-meta">Waveform, Spectrogram, Pitch & RMS Trajectories</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        sig_tab1, sig_tab2, sig_tab3 = st.tabs(["Time-Domain Waveform", "Mel-Spectrogram Heatmap", "Pitch F0 & RMS Contours"])

        import librosa

        with sig_tab1:
            time_arr = np.linspace(0, len(y_wave) / sr_val, num=len(y_wave))
            downsample = max(1, len(y_wave) // 2000)
            fig_w = go.Figure()
            fig_w.add_trace(
                go.Scatter(
                    x=time_arr[::downsample],
                    y=y_wave[::downsample],
                    mode="lines",
                    line=dict(color="#38BDF8", width=1.4),
                    name="Amplitude",
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.12)",
                )
            )
            fig_w.update_layout(
                xaxis=dict(
                    title="Time (seconds)",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.06)",
                    tickfont=dict(size=9.5, color="#94A3B8", family="JetBrains Mono, monospace"),
                    title_font=dict(size=10.5, color="#CBD5E1", family="Plus Jakarta Sans, sans-serif"),
                ),
                yaxis=dict(
                    title="Normalized Amplitude",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.06)",
                    tickfont=dict(size=9.5, color="#94A3B8", family="JetBrains Mono, monospace"),
                    title_font=dict(size=10.5, color="#CBD5E1", family="Plus Jakarta Sans, sans-serif"),
                ),
                height=230,
                margin=dict(l=35, r=20, t=15, b=25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with sig_tab2:
            S = librosa.feature.melspectrogram(y=y_wave, sr=sr_val, n_mels=64)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig_s = px.imshow(
                S_dB,
                origin="lower",
                aspect="auto",
                color_continuous_scale="Inferno",
                labels=dict(x="Time Frame", y="Mel Frequency Bin", color="dB"),
            )
            fig_s.update_layout(
                height=230,
                margin=dict(l=35, r=20, t=15, b=25),
                xaxis=dict(
                    tickfont=dict(size=9.5, color="#94A3B8", family="JetBrains Mono, monospace"),
                    title_font=dict(size=10.5, color="#CBD5E1", family="Plus Jakarta Sans, sans-serif"),
                ),
                yaxis=dict(
                    tickfont=dict(size=9.5, color="#94A3B8", family="JetBrains Mono, monospace"),
                    title_font=dict(size=10.5, color="#CBD5E1", family="Plus Jakarta Sans, sans-serif"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_s, use_container_width=True)

        with sig_tab3:
            try:
                f0 = librosa.yin(y_wave, fmin=50, fmax=500, sr=sr_val, hop_length=512)
                valid_f0 = np.where((f0 > 55) & (f0 < 490), f0, np.nan)
                rms = librosa.feature.rms(y=y_wave, hop_length=512)[0]
                times = librosa.frames_to_time(np.arange(len(valid_f0)), sr=sr_val, hop_length=512)

                fig_f = go.Figure()
                fig_f.add_trace(
                    go.Scatter(
                        x=times,
                        y=valid_f0,
                        mode="markers+lines",
                        name="Pitch F0 (Hz)",
                        line=dict(color="#FBBF24", width=1.6),
                        marker=dict(size=3, color="#F59E0B"),
                    )
                )
                fig_f.add_trace(
                    go.Scatter(
                        x=times,
                        y=rms * 450,
                        mode="lines",
                        name="RMS Energy (scaled)",
                        line=dict(color="#06B6D4", width=1.5),
                    )
                )
                fig_f.update_layout(
                    xaxis=dict(
                        title="Time (seconds)",
                        showgrid=True,
                        gridcolor="rgba(255, 255, 255, 0.06)",
                        tickfont=dict(size=9.5, color="#94A3B8", family="JetBrains Mono, monospace"),
                        title_font=dict(size=10.5, color="#CBD5E1", family="Plus Jakarta Sans, sans-serif"),
                    ),
                    yaxis=dict(
                        title="Frequency (Hz)",
                        showgrid=True,
                        gridcolor="rgba(255, 255, 255, 0.06)",
                        tickfont=dict(size=9.5, color="#94A3B8", family="JetBrains Mono, monospace"),
                        title_font=dict(size=10.5, color="#CBD5E1", family="Plus Jakarta Sans, sans-serif"),
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=9.5, color="#E2E8F0", family="Plus Jakarta Sans, sans-serif"),
                    ),
                    height=230,
                    margin=dict(l=35, r=20, t=25, b=25),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_f, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute pitch contours: {e}")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 2: PERFORMANCE & BENCHMARKS
# =========================================================
with view_eval:
    st.markdown('<div class="card-title" style="margin-bottom: 0.95rem; color: #A5B4FC;">Empirical Evaluation & Performance Metrics</div>', unsafe_allow_html=True)

    # Executive Stat Tiles with Gradient Tops
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="stat-tile stat-tile-emerald"><div class="stat-val" style="color: #34D399; text-shadow: 0 0 15px rgba(52, 211, 153, 0.3);">43.75%</div><div class="stat-lbl">SER Test Accuracy</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="stat-tile stat-tile-blue"><div class="stat-val" style="color: #60A5FA; text-shadow: 0 0 15px rgba(96, 165, 250, 0.3);">41.65%</div><div class="stat-lbl">SER Macro F1</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="stat-tile stat-tile-purple"><div class="stat-val" style="color: #A78BFA; text-shadow: 0 0 15px rgba(167, 139, 250, 0.3);">1.04%</div><div class="stat-lbl">ASR Word Error Rate</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="stat-tile stat-tile-amber"><div class="stat-val" style="color: #FBBF24; text-shadow: 0 0 15px rgba(251, 191, 36, 0.3);">98.96%</div><div class="stat-lbl">ASR Word Accuracy</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    col_cm, col_asr_bench = st.columns([1, 1])

    with col_cm:
        st.markdown(
            """
            <div class="card-shell" style="border-top: 3px solid #F59E0B;">
                <div class="card-header">
                    <span class="card-title">SER Confusion Matrix (Actors 21-24)</span>
                    <span class="card-meta">240 Held-Out Samples (Zero Leakage)</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if EVAL_REPORT_FILE.exists():
            with open(EVAL_REPORT_FILE, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
            cm_matrix = rep_data.get("confusion_matrix", [])
            class_names = rep_data.get("classes", EMOTION_LABELS)
            if cm_matrix and class_names:
                fig_cm = px.imshow(
                    cm_matrix,
                    x=class_names,
                    y=class_names,
                    color_continuous_scale="YlOrRd",
                    text_auto=True,
                    labels=dict(x="Predicted Emotion", y="Ground Truth Emotion", color="Samples"),
                )
                fig_cm.update_layout(
                    height=360,
                    margin=dict(l=35, r=20, t=10, b=25),
                    xaxis=dict(
                        tickfont=dict(size=9.5, color="#E2E8F0", family="Plus Jakarta Sans, sans-serif"),
                        title_font=dict(size=10.5, color="#A5B4FC", family="Plus Jakarta Sans, sans-serif"),
                    ),
                    yaxis=dict(
                        tickfont=dict(size=9.5, color="#E2E8F0", family="Plus Jakarta Sans, sans-serif"),
                        title_font=dict(size=10.5, color="#A5B4FC", family="Plus Jakarta Sans, sans-serif"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Evaluated on 240 held-out speech recordings across 4 unseen actors (2 male, 2 female).")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_asr_bench:
        st.markdown(
            """
            <div class="card-shell" style="border-top: 3px solid #06B6D4;">
                <div class="card-header">
                    <span class="card-title">ASR Benchmark Comparison</span>
                    <span class="card-meta">Multi-Speaker RAVDESS Evaluation</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if ASR_REPORT_FILE.exists():
            with open(ASR_REPORT_FILE, "r", encoding="utf-8") as f:
                asr_report = json.load(f)
            
            comp_list = []
            if "models" in asr_report:
                for mod_name, mod_info in asr_report["models"].items():
                    comp_list.append({
                        "Model Architecture": mod_name,
                        "WER": f"{mod_info.get('word_error_rate_wer', 0.0)*100:.2f}%",
                        "CER": f"{mod_info.get('character_error_rate_cer', 0.0)*100:.2f}%",
                        "Word Accuracy": f"{mod_info.get('accuracy_word_level', 0.0):.2f}%",
                    })
                st.dataframe(pd.DataFrame(comp_list), use_container_width=True, hide_index=True)
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-size: 0.83rem; color: #94A3B8; line-height: 1.6;">
                    <strong style="color: #F1F5F9;">Evaluation Methodology & Standardized Statements</strong><br>
                    &bull; Evaluated via <code>jiwer</code> with text normalization (lowercasing, punctuation stripped).<br>
                    &bull; Statement 1: <em>"Kids are talking by the door."</em><br>
                    &bull; Statement 2: <em>"Dogs are sitting by the door."</em><br>
                    &bull; <strong style="color: #E2E8F0;">Scope Note:</strong> Standardized read speech yields low error rates (~1-2% WER); noisy, spontaneous conversational speech will exhibit higher error rates.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 3: ARCHITECTURE & 61 FEATURES
# =========================================================
with view_arch:
    st.markdown('<div class="card-title" style="margin-bottom: 0.95rem; color: #A5B4FC;">System Architecture & Acoustic Engineering</div>', unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown(
            """
            <div class="card-shell" style="border-top: 3px solid #F97316;">
                <div class="card-header">
                    <span class="card-title">61 Acoustic Feature Vector</span>
                    <span class="card-meta">Deterministic 1D Representation</span>
                </div>
                <table style="width:100%; font-size: 0.83rem; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.12);">
                        <th style="text-align:left; padding: 7px 0; color: #A5B4FC;">Feature Category</th>
                        <th style="text-align:right; padding: 7px 0; color: #A5B4FC;">Dimensions</th>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;"><span class="tag-amber" style="margin-right:6px;">MFCC</span> Means (Coefficients 1-13)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">13</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;"><span class="tag-amber" style="margin-right:6px;">MFCC</span> Standard Deviations (Coefficients 1-13)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">13</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;"><span class="tag-blue" style="margin-right:6px;">Delta</span> MFCC Means (First-order derivative)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">13</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;"><span class="tag-blue" style="margin-right:6px;">Delta</span> MFCC Standard Deviations</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">13</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;"><span class="tag-emerald" style="margin-right:6px;">Pitch</span> (F0) Statistics (Mean, Std, Min, Max via YIN)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">4</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;"><span class="tag-purple" style="margin-right:6px;">Energy</span> RMS Energy Envelope (Mean, Std, Max)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">3</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.12);">
                        <td style="padding: 6px 0;"><span class="tag-cyan" style="margin-right:6px;">Centroid</span> Spectral Centroid (Mean, Std)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">2</td>
                    </tr>
                    <tr>
                        <td style="padding: 9px 0; font-weight: 800; color: #FFFFFF;">Total Feature Vector Dimension</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 800; color: #A5B4FC; font-size: 0.95rem;">61</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_a2:
        st.markdown(
            """
            <div class="card-shell" style="border-top: 3px solid #8B5CF6;">
                <div class="card-header">
                    <span class="card-title">Empirical Hardware Latency Profile</span>
                    <span class="card-meta">Intel CPU, Windows 11 / macOS POSIX</span>
                </div>
                <table style="width:100%; font-size: 0.83rem; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.12);">
                        <th style="text-align:left; padding: 7px 0; color: #A5B4FC;">Subroutine</th>
                        <th style="text-align:right; padding: 7px 0; color: #A5B4FC;">Warm Duration</th>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;">Audio Loading & Resampling (48k to 16k)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">~4.31 s</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;">61 Acoustic Feature Extraction</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">~186.9 ms</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;">Pure SER Inference (HistGradientBoosting)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">~130.8 ms</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;">Pure SER Inference (Random Forest)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">~87.9 ms</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;">Pure ASR Inference (Wav2Vec 2.0 CTC)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">~342.0 ms</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 6px 0;">Pure ASR Inference (Whisper Base)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F1F5F9;">~1.34 s</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.12);">
                        <td style="padding: 6px 0; font-weight: 700; color: #06B6D4;">End-to-End Turnaround (Wav2Vec2 + HGB)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 800; color: #06B6D4;">~4.97 s</td>
                    </tr>
                    <tr>
                        <td style="padding: 7px 0; font-weight: 700; color: #F59E0B;">End-to-End Turnaround (Whisper + HGB)</td>
                        <td style="text-align:right; font-family: 'JetBrains Mono', monospace; font-weight: 800; color: #F59E0B;">~5.96 s</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card-shell" style="border-top: 3px solid #6366F1;">
            <div class="card-header">
                <span class="card-title">Experimental Methodology & Actor-Independent Validation</span>
                <span class="card-meta">Strict Generalization Guarantee</span>
            </div>
            <p style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6; margin: 0;">
                The Speech Emotion Recognition subsystem is trained and validated on the official speech partition of the 
                <strong style="color: #F1F5F9;">Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)</strong>, consisting of exactly 
                <strong style="color: #F1F5F9;">1,440 speech recordings</strong> across 24 professional actors (12 male, 12 female). 
                To ensure authentic generalization to unseen speakers, an <strong style="color: #F1F5F9;">Actor-Independent Split</strong> was enforced:
                Actors 01 through 20 (1,200 recordings) comprise the training partition, while Actors 21 through 24 (240 recordings) 
                form the held-out test partition. The StandardScaler is fitted strictly on the training partition. 
                Intersection of training and testing speaker IDs is strictly null (&empty;), guaranteeing zero identity leakage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
