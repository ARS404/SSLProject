import torch
import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, in_channels, num_labels, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.num_labels = num_labels

        self.model = nn.Sequential(
            nn.ConvTranspose2d( # 16
                in_channels,
                self.hidden_dim,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose2d( # 32
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose2d( # 64
                self.hidden_dim,
                self.num_labels,
                kernel_size=4,
                stride=2
            ),
        )

    def forward(self, embeddings):
        out = self.model(embeddings)
        out = nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out
    

    def get_name(self):
        return "Upsample"
