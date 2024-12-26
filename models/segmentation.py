import torch.nn as nn

from hydra.utils import instantiate


class Segmentation(nn.Module):
    def __init__(self, backbone_conf, head_conf):
        super().__init__()
        
        self.backbone = instantiate(backbone_conf)
        self.head = instantiate(head_conf)

    def forward(self, images):
        out = self.backbone(images)
        out = self.head(out)
        return out
