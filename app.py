from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
torch.set_num_threads(1)
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
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

NUM_CLASSES = 2
CLASS_NAMES = ["Real", "AI"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "efficientnet_ai_real_1k.pth")
MODEL_VARIANT = "efficientnet-b0"

device = torch.device("cpu")
log.info(f"Using device: {device}")

def build_model(num_classes=NUM_CLASSES):
    model = EfficientNet.from_name(MODEL_VARIANT)
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def load_state_dict_strict_safe(model, state_path):
    if not os.path.exists(state_path):
        log.error(f"Model file not found at {state_path}")
        raise SystemExit(1)
    raw_state = torch.load(state_path, map_location="cpu")
    if isinstance(raw_state, torch.nn.Module):
        log.warning("Loaded full model object")
        return raw_state
    new_state = OrderedDict()
    for k, v in raw_state.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state[new_key] = v
    try:
        model.load_state_dict(new_state)
        log.info("state_dict loaded strict=True")
    except RuntimeError as e:
        log.warning("Strict load failed: %s", e)
        model.load_state_dict(new_state, strict=False)
    return model

model = build_model(NUM_CLASSES)
model = load_state_dict_strict_safe(model, MODEL_PATH)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "Veritas backend is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        file = request.files["file"]
        img_bytes = file.read()
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy().tolist()
            top_idx = int(torch.argmax(outputs, dim=1).item())
            top_prob = float(probs[top_idx])

        flipped_idx = 1 - top_idx
        flipped_prob = probs[flipped_idx]
        label = CLASS_NAMES[flipped_idx]

        return jsonify({
            "class_index": flipped_idx,
            "class_label": label,
            "probability": flipped_prob,
            "probs": probs
        })
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
