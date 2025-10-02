import os
import torch

# -----------------------------
# Model path (same folder as config.py / app.py)
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "efficientnet_faces.pth")
MODEL_VARIANT = "efficientnet-b0"

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
