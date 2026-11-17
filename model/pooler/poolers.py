from .basepooler import BasePooler
import torch

class IdentityPooler(BasePooler):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class AvgPooler(BasePooler):
    """ [Batch, Features, Time] pooling time in averaging -> [Batch, Features]"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)

class MaxPooler(BasePooler):
    """[Batch, Features, Time] pooling time in keeping maximum -> [Batch, Features]"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=-1)[0]

class SpatioTemporalAvgPooler(BasePooler):
    """[Batch, Channels, Patches, HiddenSize] -> [Batch, HiddenSize]"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2))