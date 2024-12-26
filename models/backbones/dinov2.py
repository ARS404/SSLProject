import torch
import torch.nn as nn


class DinoV2(nn.Module):
    def __init__(self, freeze, reshape):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').cuda()

        self.reshape = reshape
        self.freeze = freeze

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images):
        return self.model.get_intermediate_layers(images, 24, reshape=self.reshape)[0]
    
    def get_name(self):
        return f"DinoV2 {'(frozen)' if self.freeze else ''}"