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
        out = self.model.get_intermediate_layers(images, [20, 21, 22, 23], reshape=self.reshape)
        out = torch.cat(out, dim=1)
        print(out.shape)
        return out
    
    def get_name(self):
        return f"DinoV2 {'(frozen)' if self.freeze else ''}"