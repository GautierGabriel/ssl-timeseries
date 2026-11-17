import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseMethod(nn.Module, ABC):
    def __init__(self, target_channels: list[str] = None, normalizers: nn.ModuleDict = None):
        super().__init__()
        self.target_channels = target_channels or ["raw"]
        self.normalizers = normalizers or nn.ModuleDict()

    def on_epoch_start(self, current_epoch: int, total_epochs: int):
        pass

    def _prepare_input_tensor(self, batch_x: torch.Tensor, stats: dict = None) -> tuple[torch.Tensor, torch.Tensor]:
        if stats is None:
            stats = {"extra_channels": {}}

        dynamic_tensors = []
        static_tensors = []

        for ch in self.target_channels:
            x_ch = batch_x if ch == "raw" else stats["extra_channels"].get(ch, batch_x)

            if ch in self.normalizers:
                result = self.normalizers[ch](x_ch, stats)
                if isinstance(result, tuple) and len(result) == 2:
                    dyn, stat = result
                    static_tensors.append(stat)
                else:
                    dyn = result[0] if isinstance(result, tuple) else result
            else:
                dyn = x_ch

            if dyn.ndim == 2:
                dyn = dyn.unsqueeze(1)
            dynamic_tensors.append(dyn)

        x_dynamic = torch.cat(dynamic_tensors, dim=1)

        if static_tensors:
            x_static = torch.cat(static_tensors, dim=1)
        else:
            x_static = torch.empty((batch_x.size(0), 0), device=batch_x.device)

        return x_dynamic, x_static

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    def get_projection(self, x: torch.Tensor, stats: dict = None, **kwargs) -> torch.Tensor:
        x_dyn, x_stat = self._prepare_input_tensor(x, stats)

        if getattr(self, "input_chunk_length", None) is not None:
            x_dyn = x_dyn[:, :, :self.input_chunk_length]

        latent = self.encoder(x_dyn, x_stat)

        if getattr(self, "pooler", None) is not None:
            latent = self.pooler(latent)

        return latent

    def get_prediction(self, x: torch.Tensor, stats: dict = None) -> torch.Tensor:
        latent = self.get_projection(x, stats)

        prediction = self.head(latent)

        return prediction

    def compute_training_metrics(self, batch_x: torch.Tensor, stats: dict = None) -> dict:
        return {}

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)