import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# Recreate architecture
num_classes = 2  # same as training
model = EfficientNet.from_name('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, num_classes)

# Load weights
state_dict = torch.load("efficientnet_ai_real_1k.pth", map_location="cpu")
model.load_state_dict(state_dict)

model.eval()
