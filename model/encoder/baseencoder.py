from abc import ABC, abstractmethod
import torch
from torch import nn

class BaseEncoder(nn.Module, ABC):
    ''' Abstract base class for encoder'''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """return latent space"""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """dim of the latent space"""
        pass
