import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Base(nn.Module):
    def __init__(
        self,
        backbone: AutoModel,
        freeze_backbone: bool | str = True,
    ):
        """
        Simple regression head on top of DINOv3 backbone.
        Args:
            backbone: Pretrained DINOv3 model from HuggingFace transformers.
            freeze_backbone: If True, freeze all backbone parameters. If "attentions", freeze all except attention layers.
        """
        super().__init__()
        self.backbone = backbone

        if freeze_backbone is True:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        elif freeze_backbone in ["attentions", "attention"]:
            for name, param in self.backbone.named_parameters():
                if "attention" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.backbone.eval()
        elif not freeze_backbone:
            pass  # do not freeze anything
        else:
            raise ValueError(
                f"freeze_backbone must be bool or 'attentions', got {freeze_backbone}"
            )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a base class. Use a subclass instead.")