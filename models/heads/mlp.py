import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, tokenW, tokenH, num_labels, hidden_dim, h=224, w=224):
        super().__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.h = h
        self.w = w
        self.num_labels = num_labels

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                (1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                num_labels,
                (1, 1)
            )
        )

    def forward(self, embeddings):
        out = self.model(embeddings)
        out = nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out
    

    def get_name(self):
        return "MLP Classifier"
