import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.mobilenet_v3_small(weights=None)

        self.backbone = backbone.features

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.projection = nn.Sequential(
            nn.Linear(576,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

    def forward(self,x):

        x = self.backbone(x)

        x = self.pool(x)

        x = x.view(x.size(0),-1)

        embedding = self.projection(x)

        return embedding

if __name__ == "__main__":
    print("model.py executed successfully! Encoder class is defined and ready to be imported.") 