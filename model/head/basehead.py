import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class BaseHead(nn.Module, ABC):
    """take the latant space as input and output the right format for the pretext task"""
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass
