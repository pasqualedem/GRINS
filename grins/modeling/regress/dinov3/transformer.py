import torch
import torch.nn as nn
from transformers import AutoModel

from .base import DINOv3Base


class DINOv3Transformer(DINOv3Base):
    def __init__(
        self,
        backbone: AutoModel,
        num_tasks: int,
        num_transformer_layers: int = 1,
        activation: str = "ReLU",
        freeze_backbone: bool = True,
    ):
        """
        Simple regression head on top of DINOv3 backbone.
        Args:
            backbone: Pretrained DINOv3 model from HuggingFace transformers.
            num_tasks: Number of regression tasks (output size).
            num_layers: Number of linear layers in the head (default: 1 for linear probing).
            activation: Activation function to use between head layers.
            freeze_backbone: If True, freeze backbone parameters.
        """
        assert num_transformer_layers >= 1, "num_transformer_layers must be >= 1"
        assert num_tasks >= 1, "num_tasks must be >= 1"
        super().__init__(
            backbone=backbone,
            freeze_backbone=freeze_backbone,
        )

        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Backbone model does not have 'hidden_size' attribute in its config."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            activation=activation.lower(),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.head = nn.Linear(hidden_size * 2, num_tasks)

    def forward(self, *args, **kwargs):
        outputs = self.backbone(*args, **kwargs)
        transformer_input = outputs.last_hidden_state  # [B, P, C]
        outputs = self.transformer(transformer_input)  # [B, P-4, C]
        pooled = outputs[:, 0, :]  # [B, C]
        patch_outputs = outputs[:, 5:, :]  # [B, P-5, C]
        avg_patch_outputs = patch_outputs.mean(dim=1)  # [B, C]
        features = torch.cat([pooled, avg_patch_outputs], dim=-1)  # [B, 2*C]
        return self.head(features)
