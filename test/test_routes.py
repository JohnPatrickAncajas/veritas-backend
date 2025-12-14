"""Simple helpers to call the backend routes with a local image file.

Provides two functions:
- predict(image_path): POSTs to /predict and prints the JSON response
- detect_face(image_path): POSTs to /detect_face and prints the JSON response

To run the back-end server locally:
    > $env:PORT=5000; python -m waitress --port=5000 app:app
"""

import os
import json
import re
import base64
from pathlib import Path
import requests
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import torch
from typing import Any, cast

BASE = os.environ.get("BASE_URL", "http://localhost:5000")
 
def main() -> None:
    # Example: Crop faces from a folder using MTCNN
    input_folder = r"D:\Downloads\celebs"
    output_folder = r"D:\Downloads\celebs_cropped"
    
    
    # Try with lower confidence threshold first
    crop_faces_combined(input_folder, output_folder, mtcnn_conf_thresh=0.7, yolov5_conf_thresh=0.1)
    # crop_faces_mtcnn(input_folder, output_folder, conf_thresh=0.7)

def ok_get_root() -> None:
    r = requests.get(f"{BASE}/")
    print("GET / ->", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)


def predict(image_path: str) -> None:
    """POST image to /predict and print JSON response."""
    if not os.path.isfile(image_path):
        print(f"predict: file not found: {image_path}")
        return
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        r = requests.post(f"{BASE}/predict", files=files)
    print("POST /predict ->", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)


def detect_face(image_path: str) -> None:
    """POST image to /detect_face and print JSON response."""
    if not os.path.isfile(image_path):
        print(f"detect_face: file not found: {image_path}")
        return
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        r = requests.post(f"{BASE}/detect_face", files=files)
    print("POST /detect_face ->", r.status_code)
    try:
        resp_json = r.json()
        print(json.dumps(resp_json, indent=2))
        return resp_json
    except Exception:
        print(r.text)
        return None


def save_crops_from_response(resp_json: dict, out_dir: str = "images") -> list:
    """Save any base64 'crop' images found in the detection response to out_dir.

    Saves files into a folder next to this test script and returns list of file paths.
    """
    saved = []
    if not resp_json:
        return saved
    detections = resp_json.get("detections", [])
    base = Path(__file__).resolve().parent
    images_dir = base.joinpath(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    data_url_re = re.compile(r"^data:(image/[^;]+);base64,(.+)$")
    for i, d in enumerate(detections):
        data_url = d.get("crop")
        if not data_url:
            continue
        m = data_url_re.match(data_url)
        if not m:
            continue
        mime, b64 = m.group(1), m.group(2)
        ext = "jpg" if "jpeg" in mime or "jpg" in mime else mime.split("/")[-1]
        filename = f"crop_{i}.{ext}"
        out_path = images_dir.joinpath(filename)
        with open(out_path, "wb") as fh:
            fh.write(base64.b64decode(b64))
        saved.append(str(out_path))
    return saved


def crop_faces_mtcnn(input_folder: str, output_folder: str, conf_thresh: float = 0.9) -> None:
    """Crop human faces from all images in input_folder using MTCNN and save to output_folder.
    
    Args:
        input_folder: Path to folder containing images to process
        output_folder: Path to folder where cropped face images will be saved
        conf_thresh: Minimum confidence threshold for face detection (default: 0.9)
    
    Returns:
        None. Prints progress and saves cropped faces to output_folder.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        return
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize MTCNN detector
    print("Initializing MTCNN face detector...")
    detector = MTCNN()
    
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    
    # Get all image files from input folder
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    total_faces_saved = 0
    
    for img_file in image_files:
        try:
            # Load image
            pil_img = Image.open(img_file).convert("RGB")
            np_img = np.array(pil_img)
            
            # Detect faces
            results = detector.detect_faces(np_img)
            
            if not results:
                print(f"  {img_file.name}: No faces detected")
                continue
            
            # Filter by confidence and crop each face
            faces_in_image = 0
            for face_idx, r in enumerate(results):
                confidence = float(r.get("confidence", 0.0))
                if confidence < conf_thresh:
                    continue
                
                # Extract bounding box (MTCNN returns [x, y, width, height])
                x, y, w, h = r["box"]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                # Ensure bounds are within image
                img_w, img_h = pil_img.size
                x1 = max(0, min(x1, img_w - 1))
                x2 = max(0, min(x2, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop face
                face_crop = pil_img.crop((x1, y1, x2, y2))
                
                # Generate output filename
                stem = img_file.stem
                output_filename = f"{stem}_face{face_idx}.jpg"
                output_file_path = output_path / output_filename
                
                # Save cropped face
                face_crop.save(output_file_path, "JPEG", quality=95)
                faces_in_image += 1
                total_faces_saved += 1
            
            if faces_in_image > 0:
                print(f"  {img_file.name}: Saved {faces_in_image} face(s)")
            else:
                print(f"  {img_file.name}: No faces above confidence threshold ({conf_thresh})")
                
        except Exception as e:
            print(f"  {img_file.name}: Error - {e}")
    
    print(f"\nCompleted! Total faces saved: {total_faces_saved}")
    print(f"Output folder: {output_folder}")


def crop_faces_yolov5_2d(input_folder: str, output_folder: str, conf_thresh: float = 0.4) -> None:
    """Crop 2D anime faces from all images in input_folder using YOLOv5 and save to output_folder.
    
    Args:
        input_folder: Path to folder containing images to process
        output_folder: Path to folder where cropped face images will be saved
        conf_thresh: Minimum confidence threshold for face detection (default: 0.4)
    
    Returns:
        None. Prints progress and saves cropped faces to output_folder.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        return
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize YOLOv5 2D face detector
    print("Initializing YOLOv5 2D anime face detector...")
    
    # Get the backend project root (where app.py is located)
    backend_root = Path(__file__).resolve().parent.parent
    face_model_path = backend_root / "models" / "yolov5s_anime.pt"
    yolov5_dir = backend_root / "vendor" / "yolov5"
    
    if not face_model_path.exists():
        print(f"Error: Face model not found at {face_model_path}")
        return
    
    if not yolov5_dir.exists():
        print(f"Error: YOLOv5 repository not found at {yolov5_dir}")
        return
    
    # Load YOLOv5 model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.hub.load(
            str(yolov5_dir.resolve()),
            "custom",
            path=str(face_model_path),
            source="local",
            force_reload=False
        )
        model.to(device)
        model.eval()
        names = model.names if hasattr(model, "names") else {}
        meta = {"type": "yolov5", "names": names}
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return
    
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    
    # Get all image files from input folder
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    total_faces_saved = 0
    
    for img_file in image_files:
        try:
            # Load image
            pil_img = Image.open(img_file).convert("RGB")
            np_img = np.array(pil_img)
            
            # Detect faces
            results = model(np_img)
            
            # Parse results (YOLOv5 format)
            detections = _parse_yolov5_results(results, meta, conf_thresh)
            
            # Debug: show all detections before filtering
            all_detections = _parse_yolov5_results(results, meta, conf_thresh=0.0)
            if all_detections and not detections:
                max_conf = max(d["confidence"] for d in all_detections)
                print(f"  {img_file.name}: Found {len(all_detections)} detection(s) but all below threshold {conf_thresh} (max confidence: {max_conf:.3f})")
            elif not detections:
                print(f"  {img_file.name}: No faces detected")
                continue
            
            if not detections:
                continue
            
            # Crop each face
            faces_in_image = 0
            for face_idx, det in enumerate(detections):
                bbox = det["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                
                # Ensure bounds are within image
                img_w, img_h = pil_img.size
                x1 = max(0, min(x1, img_w - 1))
                x2 = max(0, min(x2, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop face
                face_crop = pil_img.crop((x1, y1, x2, y2))
                
                # Generate output filename
                stem = img_file.stem
                output_filename = f"{stem}_face{face_idx}.jpg"
                output_file_path = output_path / output_filename
                
                # Save cropped face
                face_crop.save(output_file_path, "JPEG", quality=95)
                faces_in_image += 1
                total_faces_saved += 1
            
            if faces_in_image > 0:
                print(f"  {img_file.name}: Saved {faces_in_image} face(s)")
            else:
                print(f"  {img_file.name}: No faces above confidence threshold ({conf_thresh})")
                
        except Exception as e:
            print(f"  {img_file.name}: Error - {e}")
    
    print(f"\nCompleted! Total faces saved: {total_faces_saved}")
    print(f"Output folder: {output_folder}")


def _parse_yolov5_results(results, meta, conf_thresh=0.4):
    """Parse YOLOv5 detection results and return list of detections.
    
    Returns list of dicts with bbox [x1,y1,x2,y2], confidence, class_idx, class_label.
    """
    dets = []
    results = cast(Any, results)
    
    try:
        # Try ultralytics Results path first
        if hasattr(results, "boxes") or (isinstance(results, (list, tuple)) and hasattr(results[0], "boxes")):
            r = results[0] if isinstance(results, (list, tuple)) else results
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                xy = getattr(boxes, "xyxy", None)
                confs = getattr(boxes, "conf", None)
                cls_idx = getattr(boxes, "cls", None)
                if xy is not None:
                    xy_arr = np.array(xy.cpu()) if hasattr(xy, "cpu") else np.array(xy)
                    conf_arr = np.array(confs.cpu()) if confs is not None and hasattr(confs, "cpu") else np.ones(len(xy_arr))
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
        
        # YOLOv5 hub Results path: results.xyxy[0] (tensor Nx6: x1,y1,x2,y2,conf,cls)
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
                x1, y1, x2, y2, c, ci = row[0], row[1], row[2], row[3], row[4], int(row[5])
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
    except Exception as e:
        print(f"Error parsing detection results: {e}")
    
    return dets


def crop_faces_combined(input_folder: str, output_folder: str, mtcnn_conf_thresh: float = 0.9, yolov5_conf_thresh: float = 0.1) -> None:
    """Crop faces from all images using MTCNN first, fallback to YOLOv5 if MTCNN finds nothing.
    
    Args:
        input_folder: Path to folder containing images to process
        output_folder: Path to folder where cropped face images will be saved
        mtcnn_conf_thresh: Minimum confidence threshold for MTCNN detection (default: 0.9)
        yolov5_conf_thresh: Minimum confidence threshold for YOLOv5 detection (default: 0.1)
    
    Returns:
        None. Prints progress and saves cropped faces to output_folder.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        return
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize MTCNN detector
    print("Initializing MTCNN face detector...")
    mtcnn_detector = MTCNN()
    
    # Initialize YOLOv5 2D face detector
    print("Initializing YOLOv5 2D anime face detector...")
    backend_root = Path(__file__).resolve().parent.parent
    face_model_path = backend_root / "models" / "yolov5s_anime.pt"
    yolov5_dir = backend_root / "vendor" / "yolov5"
    
    if not face_model_path.exists():
        print(f"Error: Face model not found at {face_model_path}")
        return
    
    if not yolov5_dir.exists():
        print(f"Error: YOLOv5 repository not found at {yolov5_dir}")
        return
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolov5_model = torch.hub.load(
            str(yolov5_dir.resolve()),
            "custom",
            path=str(face_model_path),
            source="local",
            force_reload=False
        )
        yolov5_model.to(device)
        yolov5_model.eval()
        names = yolov5_model.names if hasattr(yolov5_model, "names") else {}
        yolov5_meta = {"type": "yolov5", "names": names}
        print(f"YOLOv5 model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return
    
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    
    # Get all image files from input folder
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    total_faces_saved = 0
    mtcnn_count = 0
    yolov5_count = 0
    no_face_count = 0
    
    for img_file in image_files:
        try:
            # Load image
            pil_img = Image.open(img_file).convert("RGB")
            np_img = np.array(pil_img)
            
            # Try MTCNN first
            mtcnn_results = mtcnn_detector.detect_faces(np_img)
            
            # Filter MTCNN results by confidence
            valid_mtcnn = [r for r in mtcnn_results if float(r.get("confidence", 0.0)) >= mtcnn_conf_thresh]
            
            faces_in_image = 0
            detector_used = None
            
            if valid_mtcnn:
                # MTCNN found faces - use MTCNN detections
                detector_used = "MTCNN"
                for face_idx, r in enumerate(valid_mtcnn):
                    # Extract bounding box (MTCNN returns [x, y, width, height])
                    x, y, w, h = r["box"]
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    
                    # Ensure bounds are within image
                    img_w, img_h = pil_img.size
                    x1 = max(0, min(x1, img_w - 1))
                    x2 = max(0, min(x2, img_w - 1))
                    y1 = max(0, min(y1, img_h - 1))
                    y2 = max(0, min(y2, img_h - 1))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Crop face
                    face_crop = pil_img.crop((x1, y1, x2, y2))
                    
                    # Generate output filename
                    stem = img_file.stem
                    output_filename = f"{stem}_face{face_idx}.jpg"
                    output_file_path = output_path / output_filename
                    
                    # Save cropped face
                    face_crop.save(output_file_path, "JPEG", quality=95)
                    faces_in_image += 1
                    total_faces_saved += 1
                
                mtcnn_count += 1
                
            else:
                # MTCNN didn't find faces - try YOLOv5
                yolov5_results = yolov5_model(np_img)
                detections = _parse_yolov5_results(yolov5_results, yolov5_meta, yolov5_conf_thresh)
                
                if detections:
                    detector_used = "YOLOv5"
                    for face_idx, det in enumerate(detections):
                        bbox = det["bbox"]  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = bbox
                        
                        # Ensure bounds are within image
                        img_w, img_h = pil_img.size
                        x1 = max(0, min(x1, img_w - 1))
                        x2 = max(0, min(x2, img_w - 1))
                        y1 = max(0, min(y1, img_h - 1))
                        y2 = max(0, min(y2, img_h - 1))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Crop face
                        face_crop = pil_img.crop((x1, y1, x2, y2))
                        
                        # Generate output filename
                        stem = img_file.stem
                        output_filename = f"{stem}_face{face_idx}.jpg"
                        output_file_path = output_path / output_filename
                        
                        # Save cropped face
                        face_crop.save(output_file_path, "JPEG", quality=95)
                        faces_in_image += 1
                        total_faces_saved += 1
                    
                    yolov5_count += 1
            
            # Print status
            if faces_in_image > 0:
                print(f"  {img_file.name}: Saved {faces_in_image} face(s) using {detector_used}")
            else:
                print(f"  {img_file.name}: No faces detected by either detector")
                no_face_count += 1
                
        except Exception as e:
            print(f"  {img_file.name}: Error - {e}")
    
    print(f"\n{'='*60}")
    print(f"Completed! Total faces saved: {total_faces_saved}")
    print(f"  - MTCNN detected faces in: {mtcnn_count} images")
    print(f"  - YOLOv5 detected faces in: {yolov5_count} images")
    print(f"  - No faces found in: {no_face_count} images")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
