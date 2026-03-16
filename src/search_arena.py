import cv2
import torch
import os
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import Encoder
# Fixed import name: you named the file seed_embedding.py, not seed_embeddings.py
from seed_embedding import generate_seed_embeddings

# Robust Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "terrain_encoder.pth")
seed_folder = os.path.join(current_dir, "..", "Data", "Seeds")
output_folder = os.path.join(current_dir, "..", "Data", "MatchFoundForSeed")
os.makedirs(output_folder, exist_ok=True)

# Edge-friendly device mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = Encoder()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# MUST replicate preprocessing pipeline exactly like the seeds and training pipeline!
preprocess_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load seed embeddings (N, 128)
print(f"Loading seed embeddings from {seed_folder}...")
seed_embeddings = generate_seed_embeddings(seed_folder).to(device)
if seed_embeddings.shape[0] == 0:
    print("Warning: No seed embeddings found! Please run capture_seed.py first.")

# Start camera natively
cap = cv2.VideoCapture(0)

frame_count = 0
cooldown = 0  # Prevents saving 30+ frames per second of the exact same rock!

print("Starting Arena Search. Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Preprocess exactly like the seeds! Convert BGR to RGB 
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Add Center Crop to ensure Arena matching focuses on the identical center view!
    h, w = img_rgb.shape[:2]
    crop_size = min(h, w)
    start_y = h // 2 - crop_size // 2
    start_x = w // 2 - crop_size // 2
    img_rgb_cropped = img_rgb[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # Apply standard IRoC-U Albumentations logic
    img_tensor = preprocess_transform(image=img_rgb_cropped)["image"].unsqueeze(0).to(device)

    # 3. Extract incoming live embedding
    with torch.no_grad():
        frame_embedding = model(img_tensor) # Shape: (1, 128)

    # Apply EMA (Exponential Moving Average) to the current embedding to smooth jitter
    if frame_count == 0:
        smoothed_embedding = frame_embedding
    else:
        smoothed_embedding = 0.8 * smoothed_embedding + 0.2 * frame_embedding

    # 4. Vectorized Cosine Similarity 
    if seed_embeddings.shape[0] > 0:
        similarities = F.cosine_similarity(smoothed_embedding, seed_embeddings, dim=1)
        max_sim, best_idx = torch.max(similarities, dim=0)

        # 5. Strict Threshold & Cooldown Logic
        # Increased threshold significantly. 0.75 was mathematically too lenient in 128D space.
        if max_sim.item() > 0.88 and cooldown == 0:
            filename = os.path.join(output_folder, f"match_f{frame_count}_sim{max_sim.item():.2f}.jpg")
            
            # Draw a visual indicator (Green box) that a match was found for the operator
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), 10)
            cv2.putText(display_frame, f"Match: {max_sim.item():.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save the clean frame (without drawings) to disk
            cv2.imwrite(filename, frame)
            print(f"🔥 Feature detected! (Similarity: {max_sim.item():.2f}) Saved: {filename}")
            
            # Wait 30 frames (~1 sec) before saving another match to prevent massive disk spam
            cooldown = 30 
            
            cv2.imshow("Arena Search", display_frame)
        else:
            cv2.imshow("Arena Search", frame)
    else:
        cv2.imshow("Arena Search", frame)
            
    if cooldown > 0:
        cooldown -= 1

    frame_count += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()