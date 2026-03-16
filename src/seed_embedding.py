import torch
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Encoder

# Ensure strong, relative pathing over direct strings
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "terrain_encoder.pth")

# Device assignment for Jetson/Edge deployment safety
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Encoder()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Replicate the SimCLR ImageNet normalization used during training
preprocess_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def preprocess(img_path):

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {img_path}")
        
    # Crucial Fix: Convert OpenCV's direct BGR to proper RGB structure
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process through Albumentations to exactly match training conditions
    img_tensor = preprocess_transform(image=img)["image"]

    # Unsqueeze to add batch dimension
    img_tensor = img_tensor.unsqueeze(0).float()

    return img_tensor.to(device)

def generate_seed_embeddings(seed_folder):

    embeddings = []

    # Avoid processing invalid files (like .DS_Store or empty reads)
    valid_exts = {".jpg", ".jpeg", ".png"}

    for file in os.listdir(seed_folder):
        if os.path.splitext(file)[1].lower() not in valid_exts:
            continue

        path = os.path.join(seed_folder,file)

        img_tensor = preprocess(path)

        with torch.no_grad():
            emb = model(img_tensor)

        embeddings.append(emb)

    # In Active Learning math, we need an overall (N, 128) feature tensor, not a python list
    if len(embeddings) == 0:
        return torch.empty(0, 128).to(device)
        
    return torch.cat(embeddings, dim=0)

if __name__ == "__main__":
    test_seed_folder = os.path.join(current_dir, "..", "Data", "Seeds")
    
    print(f"Generating embeddings for captured seeds in: {test_seed_folder}...")
    try:
        all_embs = generate_seed_embeddings(test_seed_folder)
        
        print(f"✅ Executed Successfully!")
        print(f"Processed {all_embs.shape[0]} images from the live camera.")
        print(f"Final stacked embeddings shape matrix: {all_embs.shape}")
        if all_embs.shape[0] > 0:
            print("-" * 50)
            print(f"Embedding vector for the first seed:")
            print(f"{all_embs[0, :10]} ...")
    except Exception as e:
        print(f"Execution Error: {e}")