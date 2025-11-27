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

BASE = os.environ.get("BASE_URL", "http://localhost:5000")

def main() -> None:
    ok_get_root()
    img = r"D:\Programming\ProgrammingProjects\humanoid-classifier\2d_anime_dataset_extra\14741-65871.jpg"
    predict(img)
    resp = detect_face(img)
    if resp:
        saved = save_crops_from_response(resp)
        if saved:
            print("Saved crops:")
            for p in saved:
                print(" -", p)

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




if __name__ == "__main__":
    main()
