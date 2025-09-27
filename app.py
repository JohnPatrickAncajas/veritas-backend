from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import torchvision.transforms as transforms

# Load model
model = torch.load("efficientnet_ai_real_1k.pth", map_location="cpu")
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, 1)

    return jsonify({
        "class_index": int(top_idx),
        "probability": float(top_prob)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
