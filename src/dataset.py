import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TerrainDataset(Dataset):

    def __init__(self, folder, transform=None):
        self.folder = folder
        
        # Filter for valid image formats to avoid crashing on random files
        valid_extensions = {".jpg", ".jpeg", ".png"}
        self.images = [
            f for f in os.listdir(folder) 
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        # Robust SimCLR Augmentation Pipeline specifically tailored for terrain/UGVs
        # 1. Random Resized Crop: Enforces scale invariance (far rocks vs close rocks).
        # 2. Color Jitter: Super important for shadows/outdoor lighting (forces texture learning).
        # 3. Grayscale (20%): Prevents the model from relying entirely on terrain color.
        # 4. Blur (50%): Makes representation robust to out-of-focus rover camera feeds.
        self.transform = transform if transform else A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
            A.ToGray(p=0.2), 
            A.GaussianBlur(blur_limit=(3, 7), p=0.5), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.folder, self.images[index])

        # Safer read with fallback to the next image to avoid DataLoader crashes 
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((index + 1) % len(self.images))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # SimCLR requires TWO differently augmented views of the same anchor image
        aug1 = self.transform(image=image)["image"]
        aug2 = self.transform(image=image)["image"]

        # permute out image dimensions to (Channel, Height, Width) seamlessly
        return aug1, aug2

if __name__ == "__main__":
    print("dataset.py executed successfully! TerrainDataset class is defined and ready to be imported.")