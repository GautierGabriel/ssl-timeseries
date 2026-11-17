from abc import ABC, abstractmethod
import numpy as np

class BaseTransformer(ABC):
    ''' Abstract base class for Transformation of data'''
    @abstractmethod
    def __call__(self, batch_windows : np.ndarray) -> np.ndarray:
        #applied the transformation on the entire batch
        pass
