# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import io
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("veritas-backend")

app = Flask(__name__)
CORS(app)

# ------- Config (adjust if needed) -------
NUM_CLASSES = 2
CLASS_NAMES = ["Real", "AI"]  # <- update order/names if your training used different labels
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "efficientnet_ai_real_1k.pth")
MODEL_VARIANT = "efficientnet-b0"
# -----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

def build_model(num_classes=NUM_CLASSES):
    # Recreate the same architecture used during training
    model = EfficientNet.from_name(MODEL_VARIANT)
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def load_state_dict_strict_safe(model, state_path):
    # Load state_dict, handle DataParallel 'module.' prefix, and fallback to non-strict load if needed
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Model file not found: {state_path}")

    raw_state = torch.load(state_path, map_location="cpu")

    # If the saved file is a full model (not state_dict), handle it:
    if isinstance(raw_state, torch.nn.Module):
        log.warning("Loaded a full model object from file. Returning it directly.")
        return raw_state

    # raw_state should be a dict (state_dict). Normalize keys (strip 'module.' if present).
    new_state = OrderedDict()
    for k, v in raw_state.items():
        new_key = k
        if k.startswith("module."):
            new_key = k.replace("module.", "", 1)
        new_state[new_key] = v

    # Try strict load first
    try:
        model.load_state_dict(new_state)
        log.info("state_dict loaded (strict=True).")
    except RuntimeError as e:
        log.warning("Strict load_state_dict failed: %s", e)
        # Try with strict=False to inspect missing/unexpected keys
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        log.warning("Loaded with strict=False. Missing keys: %s", missing)
        log.warning("Loaded with strict=False. Unexpected keys: %s", unexpected)
    return model

# Build and load
model = build_model(NUM_CLASSES)
model = load_state_dict_strict_safe(model, MODEL_PATH)

model.to(device)
model.eval()

# Preprocessing (matches your training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)  # shape: (1,3,224,224)

        with torch.no_grad():
            outputs = model(input_tensor)  # raw logits
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy().tolist()
            top_idx = int(torch.argmax(outputs, dim=1).item())
            top_prob = float(probs[top_idx])

        label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
        return jsonify({
            "class_index": top_idx,
            "class_label": label,
            "probability": top_prob,
            "probs": probs
        })
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local testing only; Render will use Gunicorn
    app.run(host="0.0.0.0", port=5000, debug=False)
