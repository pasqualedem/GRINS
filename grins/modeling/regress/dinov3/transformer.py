import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Transformer(nn.Module):
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
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Backbone model does not have 'hidden_size' attribute in its config."
            )

        transformer_layers = []
        transformer_layers.extend(
            nn.TransformerEncoderLayer(
                d_model=hidden_size * 2,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                activation=activation.lower(),
                batch_first=True,
            )
            for _ in range(num_transformer_layers)
        )
        linear_layer = nn.Linear(hidden_size * 2, num_tasks)
        self.head = nn.Sequential(*[*transformer_layers, linear_layer])

    def forward(self, *args, **kwargs):
        outputs = self.backbone(*args, **kwargs)
        # Use pooled output if available, else first token
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        patches_last_hidden_state = outputs.last_hidden_state[:, 5:, :]  # [B, P-5, C]
        avg_patches_hidden_state = patches_last_hidden_state.mean(dim=1)  # [B, C]
        features = torch.cat([pooled, avg_patches_hidden_state], dim=-1)  # [B, 2*C]
        return self.head(features)
