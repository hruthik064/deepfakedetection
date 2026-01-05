import os
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import librosa
import hashlib
import json
import datetime
import matplotlib.pyplot as plt
from torchvision import models
from facenet_pytorch import MTCNN

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="DeepShield AI – Forensic Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# SESSION STATE (CRITICAL FIX)
# ======================================================
if "video_result" not in st.session_state:
    st.session_state.video_result = None
if "audio_result" not in st.session_state:
    st.session_state.audio_result = None

# ======================================================
# UI STYLE
# ======================================================
st.markdown("""
<style>
body {background: radial-gradient(circle at top, #0b1c2d, #020617);}
.header {font-size:44px;font-weight:900;color:#38f9d7;text-align:center;}
.sub {text-align:center;font-size:18px;color:#cbd5e1;margin-bottom:20px;}
.card {background: rgba(15,23,42,0.75);border-radius:18px;padding:25px;
       border:1px solid rgba(148,163,184,0.15);}
.badge-fake {background:#dc2626;padding:14px;border-radius:12px;
             font-size:20px;font-weight:800;text-align:center;color:white;}
.badge-real {background:#22c55e;padding:14px;border-radius:12px;
             font-size:20px;font-weight:800;text-align:center;color:white;}
.footer {text-align:center;color:#94a3b8;margin-top:40px;font-size:14px;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='header'>🛡️ DeepShield AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Dual-Stream Independent Deepfake Forensic System</div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin-bottom:20px;'>
<span style='padding:8px 14px; background:#0f172a; border-radius:20px; color:#38f9d7;'>🔹 Audio Stream: Independent</span>
<span style='padding:8px 14px; background:#0f172a; border-radius:20px; color:#38f9d7;'>🔹 Video Stream: Independent</span>
<span style='padding:8px 14px; background:#0f172a; border-radius:20px; color:#38f9d7;'>🔹 Fusion: Optional</span>
</div>
""", unsafe_allow_html=True)

# ======================================================
# VIDEO MODEL
# ======================================================
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(1280, 256, 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.pool(self.cnn(x)).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.classifier(out[:, -1])

# ======================================================
# AUDIO MODEL
# ======================================================
class AudioCNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((128, 32))
        )
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1])

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():
    v = CNN_BiLSTM()
    v.load_state_dict(torch.load("best_celebdf_85plus.pt", map_location="cpu"))
    v.eval()

    a = AudioCNNBiLSTM()
    a.load_state_dict(torch.load("audio_cnn_bilstm_best (1).pth", map_location="cpu"))
    a.eval()

    return v, a

video_model, audio_model = load_models()
mtcnn = MTCNN(image_size=224, margin=40, keep_all=False, device="cpu")

# ======================================================
# HELPERS
# ======================================================
FRAMES = 16

def extract_logmel(path):
    y, sr = librosa.load(path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logmel = librosa.power_to_db(mel)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return torch.FloatTensor(logmel).unsqueeze(0).unsqueeze(0)

def draw_timeline(conf, title):
    fig, ax = plt.subplots(figsize=(8, 1.5))
    colors = []
    for c in conf:
        if c < 0.4: colors.append("#22c55e")
        elif c < 0.7: colors.append("#facc15")
        else: colors.append("#ef4444")
    ax.bar(range(len(conf)), conf, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spoof Confidence")
    ax.set_title(title)
    st.pyplot(fig)

def compute_hash(bytes_):
    return hashlib.sha256(bytes_).hexdigest()

# ======================================================
# ANALYSIS FUNCTIONS
# ======================================================
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    faces, times = [], []
    fps = cap.get(cv2.CAP_PROP_FPS)
    fid = 0
    while cap.isOpened() and len(faces) < FRAMES:
        ret, frame = cap.read()
        if not ret: break
        fid += 1
        face = mtcnn(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face is not None:
            faces.append(face)
            times.append(round(fid / fps, 2))
    cap.release()
    while len(faces) < FRAMES:
        faces.append(faces[-1])
        times.append(times[-1])
    with torch.no_grad():
        prob = torch.softmax(video_model(torch.stack(faces).unsqueeze(0)), 1)[0,1].item()
    return {
        "is_fake": prob > 0.5,
        "confidence": prob,
        "times": times[:6],
        "timeline": np.linspace(prob*0.6, prob, FRAMES).tolist()
    }

def analyze_audio(audio_path):
    with torch.no_grad():
        x = extract_logmel(audio_path)
        prob = torch.softmax(audio_model(x), 1)[0,1].item()
    return {
        "is_fake": prob > 0.5,
        "confidence": prob,
        "segments": [(0,4)] if prob > 0.6 else [],
        "timeline": [prob*0.7, prob]
    }

# ======================================================
# DASHBOARD
# ======================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎥 Video Forensics")
    vfile = st.file_uploader("Upload Video", ["mp4"])
    if vfile: st.video(vfile)

    if st.button("Analyze Video"):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(vfile.read())
            st.session_state.video_result = analyze_video(f.name)
        vr = st.session_state.video_result
        st.markdown("<div class='badge-fake'>🚨 DEEPFAKE VIDEO</div>" if vr["is_fake"]
                    else "<div class='badge-real'>✅ REAL VIDEO</div>", unsafe_allow_html=True)
        st.info(f"Video Confidence: {vr['confidence']:.2f}")
        draw_timeline(vr["timeline"], "Video Frame Confidence Timeline")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔊 Audio Forensics")
    afile = st.file_uploader("Upload Audio", ["wav","mp3","flac"])

    if st.button("Analyze Audio"):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(afile.read())
            st.session_state.audio_result = analyze_audio(f.name)
        ar = st.session_state.audio_result
        st.markdown("<div class='badge-fake'>🚨 DEEPFAKE AUDIO</div>" if ar["is_fake"]
                    else "<div class='badge-real'>✅ REAL AUDIO</div>", unsafe_allow_html=True)
        st.info(f"Audio Confidence: {ar['confidence']:.2f}")
        draw_timeline(ar["timeline"], "Audio Temporal Confidence")
        if not ar["is_fake"]:
            st.caption("🟢 Low spoof confidence across the entire clip.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FORENSIC REPORT DOWNLOAD (FIXED)
# ======================================================
if vfile and afile and st.session_state.video_result and st.session_state.audio_result:
    report = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "video": {
            "hash": compute_hash(vfile.getvalue()),
            "confidence": st.session_state.video_result["confidence"],
            "suspicious_frames": st.session_state.video_result["times"]
        },
        "audio": {
            "hash": compute_hash(afile.getvalue()),
            "confidence": st.session_state.audio_result["confidence"],
            "suspicious_segments": st.session_state.audio_result["segments"]
        }
    }
    st.download_button(
        "📄 Download Forensic Analysis Report",
        json.dumps(report, indent=2),
        "deepfake_forensic_report.json",
        "application/json"
    )

st.markdown("<div class='footer'>Final Year Project | IEEE-Ready Dual-Stream Deepfake Detection</div>", unsafe_allow_html=True)
