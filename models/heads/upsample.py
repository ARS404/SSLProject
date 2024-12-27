import torch
import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, in_channels, num_labels, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.num_labels = num_labels

        self.upconv1 = nn.ConvTranspose2d(
            in_channels,
            self.hidden_dim,
            kernel_size=4,
            stride=2
        )
        self.upconv2 = nn.ConvTranspose2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=6,
            stride=3
        )
        self.upconv3 = nn.ConvTranspose2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=6,
            stride=3
        )
        self.upconv4 = nn.ConvTranspose2d(
            self.hidden_dim,
            self.num_labels,
            kernel_size=8,
            stride=4
        )

    def forward(self, embeddings):
        out = self.upconv1(embeddings)
        out = nn.functional.interpolate(out, size=(28, 28), mode="bilinear", align_corners=False)
        out = self.upconv2(out)
        out = nn.functional.interpolate(out, size=(56, 56), mode="bilinear", align_corners=False)
        out = self.upconv3(out)
        out = nn.functional.interpolate(out, size=(112, 112), mode="bilinear", align_corners=False)
        out = self.upconv4(out)
        out = nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out
    

    def get_name(self):
        return "Upsample"
