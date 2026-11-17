import torch
import torch.nn as nn
import torch.nn.functional as F
from .basehead import BaseHead
from vivml.SSL.model.encoder.encoders import ResidualBlock
from tsfm_public.models.tspulse import TSPulseForReconstruction

import logging

logger = logging.getLogger(__name__)

class ProjectionHead(BaseHead):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)
        return F.normalize(projected, p=2, dim=1)


class LinearClassificationHead(BaseHead):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class NonLinearClassificationHead(BaseHead):
    def __init__(self,  num_classes: int, input_dim: int, hidden_dim: int = None, dropout: float = 0.2):
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class MLPForecastingHead(BaseHead):
    def __init__(
            self,
            in_channels: int = 1,
            output_chunk_length: int = 20,
            hidden_size: int = 256,
            decoder_output_dim: int = 16,
            temporal_decoder_hidden: int = 64,
            num_decoder_layers: int = 2,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_chunk_length = output_chunk_length

        self.decoders = nn.Sequential(
            *[ResidualBlock(hidden_size, hidden_size, hidden_size, dropout, use_layer_norm) for _ in
              range(num_decoder_layers - 1)],
            ResidualBlock(hidden_size, decoder_output_dim * self.output_chunk_length, hidden_size, dropout,
                          use_layer_norm),
        )

        self.temporal_decoder = ResidualBlock(
            input_dim=decoder_output_dim,
            output_dim=self.in_channels,
            hidden_size=temporal_decoder_hidden,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        decoded = self.decoders(features)
        decoded = decoded.view(features.shape[0], self.output_chunk_length, -1)
        temporal_decoded = self.temporal_decoder(decoded)

        return temporal_decoded.transpose(1, 2)


class CosineDistanceHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("reference_mean", None)

    def fit_anchors(self, latent_reference: torch.Tensor):
        self.reference_mean = latent_reference.mean(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reference_mean is None:
            raise RuntimeError("use `fit_anchors()` to define target.")

        return F.cosine_similarity(x, self.reference_mean, dim=-1)


class MahalanobisDistanceHead(nn.Module):

    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("reference_mean", None)
        self.register_buffer("reference_cov_inv", None)

    def fit_anchors(self, latent_reference: torch.Tensor):
        self.reference_mean = latent_reference.mean(dim=0, keepdim=True)

        centered = latent_reference - self.reference_mean
        cov = (centered.T @ centered) / (latent_reference.size(0) - 1)
        cov += torch.eye(cov.size(0), device=cov.device) * self.epsilon
        self.reference_cov_inv = torch.linalg.inv(cov)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reference_mean is None:
            raise RuntimeError("use `fit_anchors()` to define target.")

        delta = x - self.reference_mean
        left = torch.matmul(delta, self.reference_cov_inv)
        distances = torch.sqrt(torch.sum(left * delta, dim=1))

        return distances


class MaskingReconstructionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_len: int,
        patch_length: int = None,
        stride: int = None,
        hidden_size: int = 48,
        num_patches: int = None,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window_len = window_len

        if num_patches is None:
            pl = patch_length if patch_length is not None else 16
            st = stride if stride is not None else 8
            num_patches = (window_len - pl) // st + 1

        self.num_patches = num_patches
        self.reconstructor = nn.Linear(hidden_size * num_patches, window_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.size(0), x.size(1), -1)
        return self.reconstructor(x_flat)


class TSPulseReconstructionHead(BaseHead):
    def __init__(self,
                 in_channels: int,
                 window_len: int = 512,
                 model_name: str = "ibm-granite/granite-timeseries-tspulse-r1",
                 revision: str = "tspulse-block-dualhead-512-p16-r1",
                 **kwargs):
        super().__init__()

        full_model = TSPulseForReconstruction.from_pretrained(
            model_name,
            revision=revision,
            num_input_channels=in_channels,
            context_length=window_len,
            mask_type="block"
        )

        self.decoder_with_head = full_model.decoder_with_head

    def forward(self, decoder_input: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> dict:

        outputs = self.decoder_with_head(
            decoder_input=decoder_input,
            loc=loc,
            scale=scale
        )
        return outputs