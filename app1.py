import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import librosa
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
# NEW UI STYLE (GLASS FORENSIC DASHBOARD)
# ======================================================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0b1c2d, #020617);
}

.header {
    font-size:44px;
    font-weight:900;
    color:#38f9d7;
    text-align:center;
}

.sub {
    text-align:center;
    font-size:18px;
    color:#cbd5e1;
    margin-bottom:30px;
}

.card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(12px);
    border-radius:18px;
    padding:25px;
    border:1px solid rgba(148,163,184,0.15);
    box-shadow: 0 0 30px rgba(0,255,209,0.08);
}

.scan-bar {
    height:6px;
    width:100%;
    background:linear-gradient(90deg, transparent, #38f9d7, transparent);
    background-size:200% 100%;
    animation: scan 1.4s linear infinite;
    border-radius:10px;
    margin:15px 0;
}

@keyframes scan {
    0% {background-position:200% 0;}
    100% {background-position:-200% 0;}
}

.badge-fake {
    background: linear-gradient(135deg,#7f1d1d,#dc2626);
    padding:18px;
    border-radius:14px;
    font-size:22px;
    font-weight:800;
    text-align:center;
    color:white;
}

.badge-real {
    background: linear-gradient(135deg,#166534,#22c55e);
    padding:18px;
    border-radius:14px;
    font-size:22px;
    font-weight:800;
    text-align:center;
    color:white;
}

.audio-chip {
    background:#7f1d1d;
    color:white;
    padding:12px 16px;
    border-radius:999px;
    font-weight:700;
    margin:6px 0;
    display:inline-block;
}

.footer {
    text-align:center;
    color:#94a3b8;
    margin-top:40px;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='header'>🛡️ DeepShield AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Dual-Stream Deepfake Forensic Detection Dashboard</div>", unsafe_allow_html=True)

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
# AUDIO MODEL (UNCHANGED)
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
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1])

# ======================================================
# LOAD MODELS (UNCHANGED)
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
# ANALYSIS FUNCTIONS (UNCHANGED)
# ======================================================
FRAMES = 16

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    faces, times = [], []
    fid, fps = 0, cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened() and len(faces) < FRAMES:
        ret, frame = cap.read()
        if not ret: break
        fid += 1
        face = mtcnn(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face is not None:
            faces.append(face)
            times.append(fid / fps)
    cap.release()
    if not faces: return False, [], []
    while len(faces) < FRAMES: faces.append(faces[-1])
    with torch.no_grad():
        probs = torch.softmax(video_model(torch.stack(faces).unsqueeze(0)), dim=1)[0]
    return probs[1] > probs[0], faces[:6], times[:6]

def extract_logmel(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logmel = librosa.power_to_db(mel)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return torch.FloatTensor(logmel).unsqueeze(0).unsqueeze(0)

def analyze_audio(audio_path):
    with torch.no_grad():
        x = extract_logmel(audio_path)
        prob = torch.softmax(audio_model(x), dim=1)[0,1].item()
    return [(0,4)] if prob > 0.6 else []

# ======================================================
# DASHBOARD UI
# ======================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎥 Video Forensics")
    vfile = st.file_uploader("Upload Video", ["mp4"])
    if vfile: st.video(vfile)
    if st.button("Analyze Video", use_container_width=True):
        st.markdown("<div class='scan-bar'></div>", unsafe_allow_html=True)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(vfile.read())
            fake, frames, times = analyze_video(f.name)
        if fake:
            st.markdown("<div class='badge-fake'>🚨 DEEPFAKE VIDEO</div>", unsafe_allow_html=True)
            cols = st.columns(len(frames))
            for i,col in enumerate(cols):
                img = frames[i].permute(1,2,0).cpu().numpy()
                col.image(np.clip(img,0,1), caption=f"{times[i]:.2f}s")
        else:
            st.markdown("<div class='badge-real'>✅ VIDEO IS REAL</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔊 Audio Forensics")

    # ✅ FLAC ADDED (ONLY CHANGE)
    afile = st.file_uploader("Upload Audio", ["wav", "mp3", "flac"])

    if st.button("Analyze Audio", use_container_width=True):
        st.markdown("<div class='scan-bar'></div>", unsafe_allow_html=True)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(afile.read())
            segs = analyze_audio(f.name)

        if segs:
            st.markdown("<div class='badge-fake'>🚨 DEEPFAKE AUDIO</div>", unsafe_allow_html=True)
            for s, e in segs:
                st.markdown(f"<div class='audio-chip'>⏱ {s}s → {e}s</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='badge-real'>✅ AUDIO IS REAL</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='footer'>Final Year Project | Dual-Stream Deepfake Detection</div>", unsafe_allow_html=True)
