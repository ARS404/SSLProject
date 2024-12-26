import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW, tokenH, num_labels, h=224, w=224):
        super().__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.num_labels = num_labels
        self.h = h
        self.w = w

        self.model = nn.Sequential(
            nn.Linear(
                self.height * self.width * self.in_channels,
                h * w * num_labels
            )
        )

    def forward(self, embeddings):
        out = embeddings.reshape(-1, self.height * self.width * self.in_channels)
        out = self.model(out)
        out = out.reshape(-1, self.num_labels, self.h, self.w)
        return out
    

    def get_name(self):
        return "Linear Classifier"
