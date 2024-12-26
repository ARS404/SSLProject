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
        out = self.model(embeddings)
        out = nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out
    

    def get_name(self):
        return "Linear Classifier"
