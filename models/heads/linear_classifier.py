import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW, tokenH, num_labels):
        super().__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.model = nn.Conv2d(
            in_channels,
            num_labels,
            (1, 1)
        )

    def forward(self, embeddings):
        out = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        out = out.permute(0, 3, 1, 2)
        out = self.model(out)
        return out
