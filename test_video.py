import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from facenet_pytorch import MTCNN

# ================= CONFIG =================
VIDEO_PATH = r"C:\Users\born2\Downloads\030.mp4"
MODEL_PATH = "best_celebdf_85plus.pt"
FRAMES = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= FACE DETECTOR =================
mtcnn = MTCNN(
    image_size=224,
    margin=40,
    min_face_size=40,
    keep_all=False,
    device=DEVICE
)

# ================= MODEL (MATCH TRAINING) =================
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.efficientnet_b0(weights=None)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.cnn(x)
        feats = self.pool(feats).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)

        lstm_out, _ = self.lstm(feats)
        return self.classifier(lstm_out[:, -1, :])

# ================= LOAD MODEL =================
model = CNN_BiLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded successfully")

# ================= VIDEO PREPROCESS =================
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
frame_read = 0
MAX_FRAMES_TO_READ = 300

while cap.isOpened() and len(frames) < FRAMES and frame_read < MAX_FRAMES_TO_READ:
    ret, frame = cap.read()
    if not ret:
        break

    frame_read += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face = mtcnn(frame)

    if face is not None:
        frames.append(face)

cap.release()

print(f"🧠 Faces detected: {len(frames)}")

# ================= FALLBACK (NO CRASH) =================
if len(frames) == 0:
    raise RuntimeError("❌ No face detected in entire video")

while len(frames) < FRAMES:
    frames.append(frames[-1])  # repeat last valid face

video_tensor = torch.stack(frames[:FRAMES]).unsqueeze(0).to(DEVICE)

# ================= INFERENCE =================
with torch.no_grad():
    logits = model(video_tensor)
    probs = torch.softmax(logits, dim=1)[0]

label = "FAKE" if probs[1] > probs[0] else "REAL"
confidence = probs.max().item() * 100

print("\n================ RESULT ================")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}%")
print("=======================================\n")
