import torch.nn as nn

from hydra.utils import instantiate


class Segmentation(nn.Module):
    def __init__(self, backbone_conf, head_conf):
        super().__init__()
        
        self.backbone = instantiate(backbone_conf)
        self.head = instantiate(
            head_conf,
            in_channels=self.backbone.get_out_channels()
        )

    def forward(self, images):
        out = self.backbone(images)
        out = self.head(out)
        return out
    
    def get_name(self):
        return f"{self.backbone.get_name()} + {self.head.get_name()}"
