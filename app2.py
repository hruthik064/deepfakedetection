import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import librosa
import time
from torchvision import models
from facenet_pytorch import MTCNN

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="DeepShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# UI STYLE
# ======================================================
st.markdown("""
<style>
body {background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);}
.title {font-size:40px;font-weight:800;color:#00FFD1;}
.sub {font-size:17px;color:#cfd8dc;}
.scan-bar {
    height:6px;width:100%;
    background:linear-gradient(90deg, transparent, #00FFD1, transparent);
    background-size:200% 100%;
    animation: scan 1.5s linear infinite;
    border-radius:5px;margin:10px 0;
}
.fake-box {background:#b91c1c;color:white;padding:20px;border-radius:12px;font-size:20px;font-weight:bold;text-align:center;}
.real-box {background:#15803d;color:white;padding:20px;border-radius:12px;font-size:20px;font-weight:bold;text-align:center;}
.audio-block {background:#7f1d1d;color:white;padding:12px;border-radius:8px;font-weight:bold;margin-bottom:8px;}
@keyframes scan {0%{background-position:200% 0;}100%{background-position:-200% 0;}}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🛡️ DeepShield AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Dual-Stream Deepfake Forensic Analysis</div>", unsafe_allow_html=True)
st.divider()

# ======================================================
# VIDEO MODEL (UNCHANGED)
# ======================================================
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(1280, 256, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.pool(self.cnn(x)).squeeze()
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.classifier(out[:, -1])

# ======================================================
# AUDIO MODEL (ASVSPOOF CNN + BiLSTM)
# ======================================================
class AudioCNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((128, 32))
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

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
    video = CNN_BiLSTM()
    video.load_state_dict(torch.load("best_celebdf_85plus.pt", map_location="cpu"))
    video.eval()

    audio = AudioCNNBiLSTM()
    audio.load_state_dict(torch.load("audio_cnn_bilstm_best (1).pth", map_location="cpu"))
    audio.eval()

    return video, audio

video_model, audio_model = load_models()

# ======================================================
# FACE DETECTOR
# ======================================================
mtcnn = MTCNN(image_size=224, margin=40, keep_all=False, device="cpu")

# ======================================================
# VIDEO ANALYSIS
# ======================================================
FRAMES = 16

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    faces, times = [], []
    fid = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened() and len(faces) < FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        face = mtcnn(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face is not None:
            faces.append(face)
            times.append(fid / fps)

    cap.release()
    if not faces:
        return False, [], []

    while len(faces) < FRAMES:
        faces.append(faces[-1])

    with torch.no_grad():
        probs = torch.softmax(video_model(torch.stack(faces).unsqueeze(0)), dim=1)[0]

    return probs[1] > probs[0], faces[:6], times[:6]

# ======================================================
# AUDIO CONFIG (ASVSPOOF)
# ======================================================
AUDIO_CFG = {
    "sr": 16000,
    "duration": 4.0,
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 128
}
TARGET_LEN = int(AUDIO_CFG["sr"] * AUDIO_CFG["duration"])

def extract_logmel(audio_path):
    y, _ = librosa.load(audio_path, sr=AUDIO_CFG["sr"])
    y = y[:TARGET_LEN] if len(y) > TARGET_LEN else np.pad(y, (0, TARGET_LEN-len(y)))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=AUDIO_CFG["sr"],
        n_fft=AUDIO_CFG["n_fft"],
        hop_length=AUDIO_CFG["hop_length"],
        n_mels=AUDIO_CFG["n_mels"]
    )
    logmel = librosa.power_to_db(mel)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return torch.FloatTensor(logmel).unsqueeze(0).unsqueeze(0)

def analyze_audio(audio_path):
    with torch.no_grad():
        x = extract_logmel(audio_path)
        out = audio_model(x)
        prob = torch.softmax(out, dim=1)[0, 1].item()
    return [(0, int(AUDIO_CFG["duration"]))] if prob > 0.6 else []

# ======================================================
# UI
# ======================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎥 Video Analysis")
    video_file = st.file_uploader("Upload Video", ["mp4"])
    if video_file:
        st.video(video_file)

    if st.button("Analyze Video"):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(video_file.read())
            is_fake, frames, times = analyze_video(f.name)

        st.markdown("<div class='scan-bar'></div>", unsafe_allow_html=True)
        if is_fake:
            st.markdown("<div class='fake-box'>🚨 DEEPFAKE VIDEO</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='real-box'>✅ REAL VIDEO</div>", unsafe_allow_html=True)

with col2:
    st.subheader("🔊 Audio Analysis")
    audio_file = st.file_uploader("Upload Audio", ["wav", "mp3"])

    if st.button("Analyze Audio"):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(audio_file.read())
            segments = analyze_audio(f.name)

        st.markdown("<div class='scan-bar'></div>", unsafe_allow_html=True)
        if segments:
            st.markdown("<div class='fake-box'>🚨 DEEPFAKE AUDIO</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='real-box'>✅ REAL AUDIO</div>", unsafe_allow_html=True)

st.divider()
st.caption("Final Year Project | Dual-Stream Deepfake Detection (ASVSpoof + Video)")
