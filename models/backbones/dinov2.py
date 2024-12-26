import torch
import torch.nn as nn


VERSION_TO_CHANNELS = {
    "dinov2_vits14_reg": 384 * 4,
    "dinov2_vitb14_reg": 768 * 4,
    "dinov2_vitl14_reg": 1024 * 4,
    "dinov2_vitg14_reg": 1536 * 4,
}

VERSION_TO_LAYERS = {
    "dinov2_vits14_reg": [8, 9, 10, 11],
    "dinov2_vitb14_reg": [8, 9, 10, 11],
    "dinov2_vitl14_reg": [20, 21, 22, 23],
    "dinov2_vitg14_reg": [36, 37, 38, 39],
}


class DinoV2(nn.Module):
    def __init__(self, freeze, reshape, version):
        super().__init__()
        self.short_v = version
        self.version = f"dinov2_vit{version}14_reg"
        self.model = torch.hub.load('facebookresearch/dinov2', self.version).cuda()

        self.reshape = reshape
        self.freeze = freeze

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images):
        out = self.model.get_intermediate_layers(
            images,
            VERSION_TO_LAYERS[self.version],
            reshape=self.reshape
        )
        out = torch.cat(out, dim=1)
        return out
    
    def get_out_channels(self):
        return VERSION_TO_CHANNELS[self.version]

    def get_name(self):
        return f"DinoV2-{self.short_v} {'(frozen)' if self.freeze else ''}"