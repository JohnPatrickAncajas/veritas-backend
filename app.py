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

# -------------------------------------------------------
# Logging setup
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("veritas-backend")

# -------------------------------------------------------
# Flask app
# -------------------------------------------------------
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB max file size

# -------------------------------------------------------
# Model config
# -------------------------------------------------------
from config import MODEL_PATH, DEVICE, MODEL_VARIANT, CLASSES as CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)
device = DEVICE
log.info(f"Using device: {device}")

# -------------------------------------------------------
# Model helper functions
# -------------------------------------------------------
def build_model(num_classes=NUM_CLASSES):
    model = EfficientNet.from_name(MODEL_VARIANT)
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def load_state_dict_safe(model, state_path):
    if not os.path.exists(state_path):
        log.error(f"Model file not found at {state_path}")
        raise SystemExit(1)

    raw_state = torch.load(state_path, map_location="cpu")
    if isinstance(raw_state, torch.nn.Module):
        log.warning("Full model object found (expected state_dict). Using directly.")
        return raw_state

    new_state = OrderedDict()
    for k, v in raw_state.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state[new_key] = v

    try:
        model.load_state_dict(new_state, strict=True)
        log.info("state_dict loaded successfully (strict=True)")
    except RuntimeError as e:
        log.warning(f"Strict load failed: {e}")
        model.load_state_dict(new_state, strict=False)
        log.info("state_dict loaded successfully (strict=False)")

    return model

# -------------------------------------------------------
# Initialize model
# -------------------------------------------------------
model = build_model(NUM_CLASSES)
model = load_state_dict_safe(model, MODEL_PATH)
model.to(device)
model.eval()
log.info("✅ Model ready for inference")

# -------------------------------------------------------
# Image preprocessing
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
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

        # Load and validate image
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400

        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs_tensor = torch.nn.functional.softmax(outputs, dim=1)[0]
            probs = probs_tensor.cpu().numpy().tolist()

            # Top-1 prediction
            top_idx = int(torch.argmax(probs_tensor).item())
            top_prob = float(probs[top_idx])
            label = CLASS_NAMES[top_idx]

        return jsonify({
            "top1": {
                "class_index": top_idx,
                "class_label": label,
                "probability": top_prob
            },
            "all_probs": {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs)}
        })

    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------
# Entry
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
