import torch.nn as nn


class mlMLP(nn.Module):
    def __init__(self, in_channels, tokenW, tokenH, num_labels, hidden_dim, h=224, w=224):
        super().__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.h = h
        self.w = w
        self.num_labels = num_labels

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.height * self.width * self.in_channels,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                h * w * num_labels
            )
        )

    def forward(self, embeddings):
        out = self.model(embeddings)
        out = out.reshape(-1, self.num_labels, self.h, self.w)
        return out
    

    def get_name(self):
        return "MLP Classifier"
