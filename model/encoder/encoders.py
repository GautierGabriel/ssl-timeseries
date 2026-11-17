import logging
import torch
import torch.nn as nn
from .baseencoder import BaseEncoder
from transformers import PatchTSTConfig, PatchTSTModel
from tsfm_public.models.tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.helpers import get_embeddings
from momentfm import MOMENTPipeline


logger = logging.getLogger(__name__)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class Conv1DEncoder(BaseEncoder):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, output_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self._output_dim = output_dim

        layers = [
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
        ]

        current_channels = hidden_channels
        for _ in range(num_layers - 2):
            next_channels = current_channels * 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, next_channels, kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm1d(next_channels),
                    nn.ReLU()
                )
            )
            current_channels = next_channels

        layers.append(
            nn.Sequential(
                nn.Conv1d(current_channels, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
        )
        self.network = nn.Sequential(*layers)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Initialized Conv1DEncoder | Depth: {num_layers} layers | Parameters: {trainable_params:,}")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor = None) -> torch.Tensor:
        return self.network(x_dynamic)

    
class Conv1DEncoderLarge(BaseEncoder):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 256, output_dim: int = 1024, num_layers: int = 5):
        super().__init__()
        self._output_dim = output_dim

        layers = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        ]

        current_channels = hidden_channels
        for i in range(num_layers - 2):
            next_channels = current_channels * 2 if i < 2 else current_channels
            layers.append(ResidualBlock1D(current_channels, next_channels, stride=2))
            current_channels = next_channels

        layers.append(nn.Conv1d(current_channels, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*layers)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Conv1DEncoder (ResNet) init | Depth: %d | Params: %d", num_layers, trainable_params)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor = None) -> torch.Tensor:
        return self.network(x_dynamic)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, dropout: float, use_layer_norm: bool):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x) + self.skip(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class MLPEncoder(BaseEncoder):
    def __init__(self, in_features: int, hidden_size: int = 256, num_encoder_layers: int = 2,
                 dropout: float = 0.1, use_layer_norm: bool = True, use_static: bool = False):
        super().__init__()
        self.in_features = in_features
        self._output_dim = hidden_size
        self.use_static = use_static

        self.encoders = nn.Sequential(
            ResidualBlock(in_features, hidden_size, hidden_size, dropout, use_layer_norm),
            *[ResidualBlock(hidden_size, hidden_size, hidden_size, dropout, use_layer_norm) for _ in
              range(num_encoder_layers - 1)],
        )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Initialized MLPEncoder | Depth: {num_encoder_layers} | Parameters: {trainable_params:,}")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor = None) -> torch.Tensor:
        B = x_dynamic.shape[0]
        x = x_dynamic.reshape(B, -1)

        if self.use_static and x_static is not None and x_static.numel() > 0:
            x = torch.cat([x, x_static], dim=1)

        return self.encoders(x)

class PatchTSTEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 window_len: int,
                 patch_length: int = 16,
                 stride: int = 16,
                 hidden_size: int = 128,
                 ibm_transfer: bool = False,
                 **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.in_channels = in_channels

        if ibm_transfer:
            self.encoder = PatchTSTModel.from_pretrained(
                "ibm/patchtst-base",
                num_input_channels=in_channels,
                context_length=window_len,
                ignore_mismatched_sizes=True
            )
            self.output_dim = self.encoder.config.d_model
        else:
            config = PatchTSTConfig(
                num_input_channels=in_channels,
                context_length=window_len,
                patch_length=patch_length,
                patch_stride=stride,
                d_model=hidden_size,
                use_cls_token=False,
                **kwargs
            )
            self.encoder = PatchTSTModel(config)
            self.output_dim = hidden_size

    def forward(self, x, stats=None):
        x_hf = x.transpose(1, 2)
        outputs = self.encoder(past_values=x_hf)
        return outputs.last_hidden_state


class MomentEncoder(BaseEncoder):
    def __init__(self,
                 in_channels: int,
                 model_name: str = "AutonLab/MOMENT-1-large",
                 freeze_backbone: bool = False,
                 **kwargs):
        super().__init__()

        self.encoder = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={
                'task_name': 'reconstruction',
                'n_channels': in_channels,
                'freeze_encoder': freeze_backbone,
                'freeze_embedder': freeze_backbone,
                'reduction': 'mean',
            }
        )
        self.encoder.init()
        self._output_dim = 1024
        #512 or small, 728 for base

        logger.info("MomentEncoder initialized.")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor = None, enable_masking: bool = False, **kwargs):
        if enable_masking:
            return self.encoder(x_enc=x_dynamic)
        outputs = self.encoder.embed(x_enc=x_dynamic, reduction="none")
        return outputs.embeddings

class TSPulseEncoder(BaseEncoder):
    def __init__(self,
                 in_channels: int,
                 window_len: int = 512,
                 model_name: str = "ibm-granite/granite-timeseries-tspulse-r1",
                 revision: str = "tspulse-block-dualhead-512-p16-r1",
                 freeze_backbone: bool = False,
                 mask_ratio: float = 0.4,
                 return_full_output: bool = False,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.return_full_output = return_full_output

        full_model = TSPulseForReconstruction.from_pretrained(
            model_name,
            revision=revision,
            num_input_channels=in_channels,
            context_length=window_len,
            mask_type="block",
            mask_ratio=mask_ratio,
            fuse_fft=True,
            fft_weight=1.0
        )

        self.backbone = full_model.backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._output_dim = self.backbone.config.d_model

        logger.info("TSPulseEncoder initialized.")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_dynamic: torch.Tensor = None, x_static: torch.Tensor = None, past_values: torch.Tensor = None,
                enable_masking: bool = False, **kwargs):
        x = x_dynamic if x_dynamic is not None else past_values

        if x is None:
            raise ValueError("Input tensor required (x_dynamic or past_values).")

        x_hf = x.transpose(1, 2)
        outputs = self.backbone(past_values=x_hf, enable_masking=enable_masking)

        if self.return_full_output:
            return outputs

        return outputs.last_hidden_state