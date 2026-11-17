from abc import ABC, abstractmethod
import torch
from torch import nn


class BasePooler(nn.Module, ABC):
    '''reduces the dimensionality of the input tensor (coming from the encoder)'''

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

