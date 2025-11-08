import os
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),    # 224x224 -> 224x224
            nn.Conv2d(8, 16, 3, stride=1, padding=1),   # 224x224
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # 224x224
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # 224x224
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 224x224
            nn.Conv2d(128, 128, 3, stride=1, padding=1),# 224x224
            nn.Conv2d(128, 128, 3, stride=1, padding=1),# 224x224
            nn.Conv2d(128, 128, 3, stride=1, padding=1) # 224x224
        )

    def forward(self, x):
        return self.features(x)
        
# Load pretrained VGG16
#model = Dummy().eval()
model = vgg16(weights=VGG16_Weights.DEFAULT).eval()

# Convert to TorchScript
scripted = torch.jit.script(model)
#scripted.save(os.path.join(models_dir, "dummy_scripted.pt"))
scripted.save(os.path.join(models_dir, "vgg16_scripted.pt"))

#print("Saved TorchScript model as models/dummy_scripted.pt")
print("Saved TorchScript model as models/vgg16_scripted.pt")
