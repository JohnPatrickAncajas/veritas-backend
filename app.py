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
import io
import base64
import numpy as np
from typing import List, Dict, Any, cast
from pathlib import Path
from mtcnn import MTCNN
from config import HUMAN_FACE_CONF_THR, HUMAN_FACE_DEVICE


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
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 2MB max file size

# -------------------------------------------------------
# Model config
# -------------------------------------------------------
from config import VERITAS_MODEL_PATH, DEVICE, VERITAS_MODEL_VARIANT, CLASSES as CLASS_NAMES, FACE_2D_MODEL_PATH, FACE_2D_MODEL_CONF_THR, DROPOUT_RATE

NUM_CLASSES = len(CLASS_NAMES)
device = DEVICE
log.info(f"Using device: {device}")

# -------------------------------------------------------
# Model helper functions
# -------------------------------------------------------
def build_model(num_classes=NUM_CLASSES):
    model = EfficientNet.from_name(VERITAS_MODEL_VARIANT)
    # Match the training architecture with dropout
    model._fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Must match DROPOUT_RATE from training
        nn.Linear(model._fc.in_features, num_classes)
    )
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
# Lazy-load model (Veritas classifier)
# -------------------------------------------------------
# Use a descriptive global name to avoid confusion with other models
veritas_model = None  # global placeholder for the EfficientNet classifier

def get_model():
    global veritas_model
    if veritas_model is None:
        log.info("🔄 Lazy-loading Veritas classifier model...")
        veritas_model = build_model(NUM_CLASSES)
        veritas_model = load_state_dict_safe(veritas_model, VERITAS_MODEL_PATH)
        veritas_model.to(device)
        veritas_model.eval()
        log.info("✅ Veritas model loaded and ready for inference")
    return veritas_model

# ------------------------------------------------------------------
# Face detector loader + parser (add near your other model helpers)
# ------------------------------------------------------------------
_face_detector = None
_face_detector_meta = {}

def get_face_detector():
    """Lazy-load the face detector model via torch.hub (ultralytics/yolov5)."""
    global _face_detector, _face_detector_meta
    if _face_detector is not None:
        return _face_detector, _face_detector_meta

    face_path = os.environ.get("FACE_2D_MODEL_PATH", None) or FACE_2D_MODEL_PATH
    device = DEVICE  # use your config constant

    if not os.path.exists(face_path):
        log.error(f"Face model file not found at {face_path}")
        raise SystemExit(1)

    log.info("Loading YOLOv5 face detector via torch.hub (ultralytics/yolov5)")
    # Load custom model from provided path. This will use the yolov5 repo implementation via torch.hub.
    repo_root = Path(__file__).resolve().parent  # directory containing app.py
    yolov5_dir = repo_root.joinpath("vendor", "yolov5")
    yolov5_path = str(yolov5_dir.resolve())

    if not yolov5_dir.exists():
        log.error(f"Local yolov5 repo not found at {yolov5_path}")
        raise SystemExit(1)

    model = torch.hub.load(
        yolov5_path,                       # absolute path to vendored repo
        "custom",
        path=face_path,                    # your .pt model file path
        source="local",                    # tells torch.hub to load from local dir
        force_reload=False
    )

    try:
        model.to(device) # type: ignore
    except Exception:
        # some hub models may not expose .to or may already be on correct device
        pass
    try:
        model.eval() # type: ignore
    except Exception:
        pass
    names = model.names if hasattr(model, "names") else {} # type: ignore
    _face_detector = cast(Any, model)
    _face_detector_meta = {"type": "yolov5", "names": names}
    return _face_detector, _face_detector_meta

_human_face_detector = None

def get_human_face_detector():
    global _human_face_detector
    if _human_face_detector is None:
        log.info("Loading MTCNN human face detector...")
        _human_face_detector = MTCNN(device=HUMAN_FACE_DEVICE)
    return _human_face_detector

def _parse_detection_results(results, meta, conf_thresh=0.4):
    """Return list of detections: dicts with bbox [x1,y1,x2,y2], confidence, class_idx, class_label."""
    dets = []
    # results comes from dynamic model backends (torch.hub / yolov5) and
    # may not have precise static types. Cast to Any for the type checker
    # so attribute accesses (like .pred or .xyxy) don't raise diagnostics.
    results = cast(Any, results)
    try:
        # ultralytics Results path (Results or sequence)
        if hasattr(results, "boxes") or (isinstance(results, (list, tuple)) and hasattr(results[0], "boxes")):
            r = results[0] if isinstance(results, (list, tuple)) else results
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                # boxes.xyxy, boxes.conf, boxes.cls (depending on ultralytics version)
                xy = getattr(boxes, "xyxy", None)
                confs = getattr(boxes, "conf", None)
                cls_idx = getattr(boxes, "cls", None)
                if xy is not None:
                    xy_arr = np.array(xy.cpu()) if hasattr(xy, "cpu") else np.array(xy)
                    conf_arr = np.array(confs.cpu()) if confs is not None and hasattr(confs, "cpu") else (np.array(conf_arr) if confs is not None else np.ones(len(xy_arr))) # type: ignore
                    cls_arr = np.array(cls_idx.cpu()).astype(int) if cls_idx is not None and hasattr(cls_idx, "cpu") else np.zeros(len(xy_arr), dtype=int)
                    names = meta.get("names", {})
                    for (x1, y1, x2, y2), c, ci in zip(xy_arr, conf_arr, cls_arr):
                        if float(c) < conf_thresh:
                            continue
                        label = names[int(ci)] if isinstance(names, (list, dict)) and int(ci) in names else str(int(ci))
                        dets.append({
                            "bbox": [int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))],
                            "confidence": float(c),
                            "class_idx": int(ci),
                            "class_label": label
                        })
                    return dets

        # yolov5 hub Results path: results.xyxy[0] (tensor Nx6: x1,y1,x2,y2,conf,cls)
        xyxy_attr = getattr(results, "xyxy", None)
        pred_attr = getattr(results, "pred", None)
        if xyxy_attr is not None or isinstance(pred_attr, list):
            xyxy = None
            if xyxy_attr is not None:
                xyxy = xyxy_attr[0]
            elif isinstance(pred_attr, list):
                xyxy = pred_attr[0]
            if xyxy is None:
                return dets
            arr = np.array(xyxy.cpu()) if hasattr(xyxy, "cpu") else np.array(xyxy)
            names = meta.get("names", {})
            for row in arr:
                x1,y1,x2,y2,c,ci = row[0],row[1],row[2],row[3],row[4],int(row[5])
                if float(c) < conf_thresh:
                    continue
                label = names[int(ci)] if isinstance(names, (list, dict)) and int(ci) in names else str(int(ci))
                dets.append({
                    "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    "confidence": float(c),
                    "class_idx": int(ci),
                    "class_label": label
                })
            return dets
    except Exception:
        log.exception("Error parsing detection results")
    return dets

# ------------------------------------------------------------------
# Crop helper
# ------------------------------------------------------------------
def crop_and_encode(pil_img: Image.Image, bbox: List[int], fmt: str = "JPEG", q: int = 90) -> str:
    """Crop PIL image by bbox [x1,y1,x2,y2] and return a base64 data URL string."""
    x1, y1, x2, y2 = bbox
    # ensure bounds inside image
    w, h = pil_img.size
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = pil_img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format=fmt, quality=q)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

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
        input_tensor = transform(image).unsqueeze(0).to(device) # type: ignore[attr-defined]

        # Lazy-load model and predict
        model_instance = get_model()
        with torch.no_grad():
            outputs = model_instance(input_tensor)
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

# ------------------------------------------------------------------
# Route: detect_face
# ------------------------------------------------------------------
@app.route("/detect_face", methods=["POST"])
def detect_face():
    """Detect faces and return boolean, count, and cropped images (base64)."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    include_crops = request.form.get("include_crops", "true").lower() not in ("0","false","no")
    conf_thresh = float(request.form.get("conf", FACE_2D_MODEL_CONF_THR))

    try:
        file = request.files["file"]
        img_bytes = file.read()

        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400

        np_img = np.array(pil_img)  # HWC RGB

        detector, meta = get_face_detector()

        # Run inference using the yolov5 hub model
        try:
            results_obj = detector(np_img)
        except Exception as e:
            log.exception("Model inference failed")
            return jsonify({"error": "Model inference failed", "detail": str(e)}), 500

        detections = _parse_detection_results(results_obj, meta, conf_thresh=conf_thresh)

        # Build response; add crop data (base64) if requested
        response_dets = []
        for d in detections:
            item = d.copy()
            if include_crops:
                data_url = crop_and_encode(pil_img, d["bbox"], fmt="JPEG", q=90)
                item["crop"] = data_url
            response_dets.append(item)

        return jsonify({
            "face_present": len(response_dets) > 0,
            "count": len(response_dets),
            "detections": response_dets
        })

    except Exception as e:
        log.exception("detect_face error")
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/detect_human_face", methods=["POST"])
def detect_human_face():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    conf_thresh = float(request.form.get("conf", HUMAN_FACE_CONF_THR))

    try:
        file = request.files["file"]
        img_bytes = file.read()

        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400

        np_img = np.array(pil_img)

        detector = get_human_face_detector()
        results = detector.detect_faces(np_img)

        detections = []
        for r in results:
            if float(r.get("confidence", 0.0)) < conf_thresh:
                continue
            x, y, w, h = r["box"]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            bbox = [x, y, x + w, y + h]
            item = {
                "bbox": bbox,
                "confidence": float(r.get("confidence", 0.0)),
                "keypoints": {k: [int(v[0]), int(v[1])] for k, v in (r.get("keypoints", {}) or {}).items()},
            }
            item["crop"] = crop_and_encode(pil_img, bbox, fmt="JPEG", q=90)
            detections.append(item)

        return jsonify({
            "human_face_present": len(detections) > 0,
            "count": len(detections),
            "detections": detections
        })

    except Exception as e:
        log.exception("detect_human_face error")
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------
# Entry
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
