import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
from config import VERITAS_MODEL_PATH, VERITAS_MODEL_VARIANT, DEVICE, CLASSES

NUM_CLASSES = len(CLASSES)
device = DEVICE

# -------------------------------------------------------
# Build model
# -------------------------------------------------------
model = EfficientNet.from_name(VERITAS_MODEL_VARIANT)
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)

# -------------------------------------------------------
# Load weights safely
# -------------------------------------------------------
raw_state = torch.load(VERITAS_MODEL_PATH, map_location="cpu")

if isinstance(raw_state, torch.nn.Module):
    print("⚠️ Full model object found (expected state_dict). Using directly.")
    model = raw_state
else:
    new_state = OrderedDict()
    for k, v in raw_state.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state[new_key] = v

    try:
        model.load_state_dict(new_state, strict=True)
        print("✅ state_dict loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"⚠️ Strict load failed: {e}")
        model.load_state_dict(new_state, strict=False)
        print("✅ state_dict loaded successfully (strict=False)")

# -------------------------------------------------------
# Eval mode
# -------------------------------------------------------
model.eval()
print("Model ready for inference with 4 classes:", CLASSES)
