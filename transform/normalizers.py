import logging
from typing import Self

import torch
import torch.nn as nn
logger = logging.getLogger(__name__)


class MadNormalize(nn.Module):
    def __init__(self, epsilon: float = 0.8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, batch_windows: torch.Tensor, *kwargs) -> torch.Tensor:
        median = torch.median(batch_windows, dim=-1, keepdim=True).values

        abs_dev = torch.abs(batch_windows - median)
        mad = torch.median(abs_dev, dim=-1, keepdim=True).values

        under_epsilon_mask = mad < self.epsilon
        if under_epsilon_mask.any():
            num_flat_windows = under_epsilon_mask.sum().item()
            logger.debug(
                f"MAD Normalization: {num_flat_windows} windows in batch had a MAD below threshold ({self.epsilon}). "
                f"Clamping applied."
            )
        mad = torch.clamp(mad, min=self.epsilon)

        return (batch_windows - median) / mad


class RunningMadNormalize(nn.Module):
    def __init__(self, channel_name, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_median", torch.tensor(0.0))
        self.register_buffer("running_mad", torch.tensor(1.0))

    def forward(self, x: torch.Tensor,*kwargs) -> torch.Tensor:
        if self.training:
            batch_median = torch.median(x)
            batch_mad = torch.median(torch.abs(x - batch_median))

            # Mise à jour lente (EMA)
            self.running_median = (1 - self.momentum) * self.running_median + self.momentum * batch_median
            self.running_mad = (1 - self.momentum) * self.running_mad + self.momentum * batch_mad

        return (x - self.running_median) / (self.running_mad + 1e-6)


class DriftNormalize(nn.Module):
    def __init__(self, channel_name: str = "raw", epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.channel_name = channel_name

    def forward(self, x: torch.Tensor, metadata: dict, *kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.squeeze(1)

        B, L = x.shape
        drift_params = metadata["drift"]
        start_idx = metadata["start_idx"]
        batch_std = metadata["stds"][self.channel_name]

        a = drift_params[:, 0].view(B, 1)
        b = drift_params[:, 1].view(B, 1)

        t = start_idx.view(B, 1) + torch.arange(L, device=x.device).view(1, L)
        drift_line = a * t + b

        safe_std = torch.clamp(batch_std.view(B, 1), min=self.epsilon)

        x_norm = ((x - drift_line) / safe_std).unsqueeze(1)

        std_log = torch.log(safe_std)
        statics = torch.cat([a, b, std_log], dim=1)

        return x_norm, statics


class HFNormalize(nn.Module):
    def __init__(self, channel_name: str = "HF", epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.channel_name = channel_name

    def forward(self, x: torch.Tensor, metadata: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.squeeze(1)

        batch_med = torch.median(x, dim=-1, keepdim=True)[0]
        x_centered = x - batch_med

        batch_std = metadata["stds"][self.channel_name]
        safe_std = torch.clamp(batch_std.view(-1, 1), min=self.epsilon)

        x_norm = (x_centered / safe_std).unsqueeze(1)

        batch_med_scaled = batch_med / safe_std
        std_log = torch.log(safe_std)
        statics = torch.cat([batch_med_scaled, std_log], dim=1)

        return x_norm, statics