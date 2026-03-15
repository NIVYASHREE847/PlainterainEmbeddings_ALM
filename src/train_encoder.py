import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import os
from dataset import TerrainDataset
from model import Encoder

# Dynamically resolve path so it works whether run from root or src/
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "..", "Data", "Frames")

dataset = TerrainDataset(dataset_path)

loader = DataLoader(dataset,
                    batch_size=32,
                    shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Encoder().to(device)

optimizer = optim.Adam(model.parameters(),lr=1e-4)

def nt_xent_loss(z1, z2, temperature=0.1):
    # Normalize features
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute cosine similarity and scale by temperature
    # logits_12 represents similarity of z1 against all in z2
    logits_12 = torch.matmul(z1, z2.T) / temperature
    logits_21 = torch.matmul(z2, z1.T) / temperature

    # Ground truth is simply the index of the main diagonal, 
    # since z1[i] corresponds positively to z2[i]
    labels = torch.arange(z1.size(0), device=z1.device)

    # Cross Entropy pushes correct pairs closer and all other pairs (negatives) apart
    loss1 = F.cross_entropy(logits_12, labels)
    loss2 = F.cross_entropy(logits_21, labels)

    return (loss1 + loss2) / 2

epochs = 20

for epoch in range(epochs):

    for img1,img2 in loader:

        img1 = img1.to(device)
        img2 = img2.to(device)

        z1 = model(img1)
        z2 = model(img2)

        loss = nt_xent_loss(z1, z2, temperature=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:",epoch,"Loss:",loss.item())

torch.save(model.state_dict(),"terrain_encoder.pth")