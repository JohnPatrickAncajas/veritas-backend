import os
import torch

# -----------------------------
# Model path (models folder)
# -----------------------------
# VERITAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/efficientnet_faces.pth")
VERITAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/efficientnet_faces_old.pth")
VERITAS_MODEL_VARIANT = "efficientnet-b0"

FACE_2D_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/yolov5s_anime.pt")
FACE_2D_MODEL_CONF_THR = 0.4

HUMAN_FACE_CONF_THR = 0.9     # you can tune this later
HUMAN_FACE_DEVICE = "CPU:0"   # or "GPU:0" if you use GPU

# -----------------------------
# Classes
# -----------------------------
CLASSES = ["2d", "3d", "ai", "real"]

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Optional: training hyperparameters (keep for reference)
# -----------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
IMG_SIZE = (224, 224)