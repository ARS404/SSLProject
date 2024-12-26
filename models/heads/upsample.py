import torch
import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, in_channels, num_labels):
        super().__init__()

        self.in_channels = in_channels
        self.num_labels = num_labels

        self.model = nn.ConvTranspose2d(
            in_channels,
            num_labels,
            kernel_size=16,
            stride=8
        )

    def forward(self, embeddings):
        out = self.model(
            embeddings, 
            output_size=torch.Size([
                embeddings.shape[0], 
                self.num_labels,
                224,
                224
            ])
        )
        # out = nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out
    

    def get_name(self):
        return "Upsample"
