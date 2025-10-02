# Random Regressor

import torch
from torch import nn


class IdentityBackbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class RandomRegressor(nn.Module):
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks

    def forward(self, pixel_values, **kwargs):
        batch_size = pixel_values.size(0)
        return torch.rand(batch_size, self.num_tasks, device=pixel_values.device)