import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """
    input: features (latent) pooled and projected,
    output: loss
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """return the loss as a scalar"""
        pass