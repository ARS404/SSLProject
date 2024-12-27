import torch
import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, in_channels, num_labels, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.num_labels = num_labels


        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=self.hidden_dim,
                kernel_size=9,
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=8,
                stride=1,
                padding="same"
            )
        )
        
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim,
                out_channels=self.num_labels,
                kernel_size=9,
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_labels,
                out_channels=self.num_labels,
                kernel_size=8,
                stride=1,
                padding="same"
            )
        )

    def forward(self, embeddings):
        out = self.block1(embeddings)
        out = nn.functional.interpolate(out, size=(112, 112), mode="bilinear", align_corners=False)
        out = self.block2(out)
        out = nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out
    

    def get_name(self):
        return "Upsample"
