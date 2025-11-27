import os
import torch

# -----------------------------
# Model path (models folder)
# -----------------------------
VERITAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/efficientnet_faces.pth")
VERITAS_MODEL_VARIANT = "efficientnet-b0"

FACE_2D_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/yolov5s_anime.pt")
FACE_2D_MODEL_CONF_THR = 0.4

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