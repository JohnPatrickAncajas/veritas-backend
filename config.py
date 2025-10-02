import os
import torch

# -----------------------------
# Project Base Path
# -----------------------------
PROJECT_PATH = r"C:\Users\Patrick\Documents\GitHub\veritas-model"

# -----------------------------
# Dataset Paths
# -----------------------------
DATA_DIR = os.path.join(PROJECT_PATH, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

PREDICT_ROOT = os.path.join(PROJECT_PATH, "predict")
PREDICT_FOLDER_NAME = "predict_friends"
PREDICT_DIR = os.path.join(PREDICT_ROOT, PREDICT_FOLDER_NAME)

# Kaggle cache
KAGGLE_CACHE = os.path.join(PROJECT_PATH, "kaggle_cache")

# Google cache (for 2D/3D datasets)
GOOGLE_CACHE = os.path.join(PROJECT_PATH, "google_cache")

# -----------------------------
# Model Paths
# -----------------------------
MODEL_DIR = os.path.join(PROJECT_PATH, "models")
MODEL_SAVE_DIR = MODEL_DIR
MODEL_NAME = "efficientnet_faces"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}.pth")

# -----------------------------
# Classes (ORDER FIXED!)
# -----------------------------
CLASSES = ["2d", "3d", "ai", "real"]

# -----------------------------
# Data Split
# -----------------------------
SPLIT_RATIO = (0.7, 0.2, 0.1)
MAX_RAW_IMAGES = 700

# -----------------------------
# Training Hyperparameters
# -----------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
IMG_SIZE = (224, 224)

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
