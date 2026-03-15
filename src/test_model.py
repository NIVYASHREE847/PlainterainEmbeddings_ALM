import os
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import Encoder

# Dynamically resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "terrain_encoder.pth")

# Grab a real frame exported from your dataset as a test sample
sample_img_path = os.path.join(current_dir, "..", "Data", "Frames", "0_Field_Grass_3840x2160_0000.jpg")

print(f"Loading model from: {model_path}")
model = Encoder()
# Map to correct device (CUDA/CPU) automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.to(device)
model.eval() # Prevent BatchNorm/Dropout from changing behavior

print(f"Loading sample image from: {sample_img_path}")
img = cv2.imread(sample_img_path)
if img is None:
    raise FileNotFoundError(f"Sample image not found at {sample_img_path}")
    
# 1. Convert OpenCV's default BGR format to standard RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Replicate standard ImageNet normalization used during training in dataset.py
test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Apply transform and add batch dimension (Batch, Channels, Height, Width)
img_tensor = test_transform(image=img)["image"].unsqueeze(0).to(device)

# 3. Extract embedding without tracking gradients for speed and memory efficiency
with torch.no_grad():
    embedding = model(img_tensor)

print("-" * 50)
print(f"✅ Success! Testing script executed successfully for IRoC-U.")
print(f"Input tensor shape: {img_tensor.shape}")
print(f"Extracted Embedding shape: {embedding.shape}")
print("-" * 50)
print(f"First 10 values of the 128-dimensional embedding vector:")
print(f"{embedding[0, :10]}")