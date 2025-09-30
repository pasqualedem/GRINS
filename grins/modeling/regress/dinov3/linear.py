import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Linear(nn.Module):
    def __init__(
        self,
        backbone: AutoModel,
        num_tasks: int,
        num_head_layers: int = 1,
        activation: str = "ReLU",
        freeze_backbone: bool | str = True,
    ):
        """
        Simple regression head on top of DINOv3 backbone.
        Args:
            backbone: Pretrained DINOv3 model from HuggingFace transformers.
            num_tasks: Number of regression tasks (output size).
            num_head_layers: Number of linear layers in the head (default: 1 for linear probing).
            activation: Activation function to use between head layers.
            freeze_backbone: If True, freeze all backbone parameters. If "attentions", freeze all except attention layers.
        """
        super().__init__()
        self.backbone = backbone

        if freeze_backbone is True:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        elif freeze_backbone == "attentions":
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

        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Backbone model does not have 'hidden_size' attribute in its config."
            )

        head_layers = []
        input_size = hidden_size * 2
        for i in range(num_head_layers):
            output_size = num_tasks if i == num_head_layers - 1 else input_size // 2
            head_layers.append(nn.Linear(input_size, output_size))
            if i < num_head_layers - 1:
                activation_layer = getattr(nn, activation)
                if activation_layer is None:
                    raise ValueError(
                        f"Activation function '{activation}' not found in torch.nn."
                    )
                head_layers.append(activation_layer())
            input_size = output_size
        self.head = nn.Sequential(*head_layers)

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
