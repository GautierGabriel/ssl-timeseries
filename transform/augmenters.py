from .basetransformer import BaseTransformer
import torch
import torch.nn as nn
import random

class TimeSeriesAugmenter(BaseTransformer):
    def __init__(
            self,
            p_noise=0.5, p_scale=0.5, p_shift=0.5, p_mask=0.5,
            noise_std=0.05,
            min_scale=0.5, max_scale=2.0,
            max_shift=50,
            mask_ratio=0.20
    ):
        self.p_noise = p_noise
        self.p_scale = p_scale
        self.p_shift = p_shift
        self.p_mask = p_mask

        self.noise_std = noise_std
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_shift = max_shift
        self.mask_ratio = mask_ratio

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.noise_std * torch.randn_like(x)

    def amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(x.size(0), x.size(1), 1, device=x.device).uniform_(self.min_scale, self.max_scale)
        return x * scale

    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (b, c, 1), device=x.device)

        idx = torch.arange(t, device=x.device).view(1, 1, t)
        shifted_idx = (idx - shifts) % t

        return torch.gather(x, dim=-1, index=shifted_idx.expand(b, c, t))

    def time_mask(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        mask_len = max(1, int(t * self.mask_ratio))
        starts = torch.randint(0, t - mask_len + 1, (b, c, 1), device=x.device)

        idx = torch.arange(t, device=x.device).view(1, 1, t)
        mask = (idx >= starts) & (idx < (starts + mask_len))

        out = x.clone()
        out = out.masked_fill(mask, 0.0)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        if random.random() < self.p_noise:
            out = self.add_noise(out)
        if random.random() < self.p_scale:
            out = self.amplitude_scale(out)
        if random.random() < self.p_shift:
            out = self.time_shift(out)
        if random.random() < self.p_mask:
            out = self.time_mask(out)
        return out


class StaticAugmenter(BaseTransformer):
    def __init__(self, noise_std: float = 0.02, p_dropout: float = 0.2):
        super().__init__()
        self.noise_std = noise_std
        self.dropout = nn.Dropout(p=p_dropout)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x

        noise = 1.0 + self.noise_std * torch.randn_like(x)
        out = x * noise

        out = self.dropout(out)

        return out


class PatchMaskingAugmenter(BaseTransformer):
    def __init__(self, mask_ratio: float = 0.40, patch_len: int = 16):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_len = patch_len

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        num_patches = t // self.patch_len

        patch_mask = torch.rand(b, c, num_patches, device=x.device) < self.mask_ratio

        mask = patch_mask.unsqueeze(-1).repeat(1, 1, 1, self.patch_len).view(b, c, -1)

        if t > mask.shape[-1]:
            padding = torch.zeros(b, c, t - mask.shape[-1], dtype=torch.bool, device=x.device)
            mask = torch.cat([mask, padding], dim=-1)

        out = x.clone()
        return out.masked_fill(mask, 0.0)