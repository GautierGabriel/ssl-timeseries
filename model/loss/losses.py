import torch
import torch.nn.functional as F
from .baseloss import BaseLoss
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class NTXentLoss(BaseLoss):
    """Compute contrastive loss on a single tensor of views."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inputs: z of shape [2 * Batch_Size, Dim]
        """
        batch_size = z.size(0) // 2

        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.temperature

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e9)

        return F.cross_entropy(sim_matrix, labels)

class WeightedCrossEntropy(BaseLoss):
    def __init__(self, weights: Optional[Union[List[float], torch.Tensor]] = None):
        super().__init__()

        if isinstance(weights, list):
            self.weights = torch.tensor(weights, dtype=torch.float32)
            logger.info(f"Poids chargés : {self.weights}")
        else:
            self.weights = weights

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.weights is not None and self.weights.device != z.device:
            self.weights = self.weights.to(z.device)

        return F.cross_entropy(z, labels, weight=self.weights)


class MSELoss(BaseLoss):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {predictions.shape} vs targets {targets.shape}")

        return F.mse_loss(predictions, targets, reduction=self.reduction)

class MAELoss(BaseLoss):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {predictions.shape} vs targets {targets.shape}")

        return F.l1_loss(predictions, targets, reduction=self.reduction)

class HuberLoss(BaseLoss):
    def __init__(self, reduction: str = 'mean', delta: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {predictions.shape} vs targets {targets.shape}")

        return F.huber_loss(predictions, targets, reduction=self.reduction, delta=self.delta)

class SimpleCrossEntropy(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(z, labels)


class MaskedMSELoss(BaseLoss):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bool_mask = mask.type(torch.bool)

        if not bool_mask.any():
            return torch.tensor(0.0, device=targets.device, requires_grad=True)

        return F.mse_loss(predictions[bool_mask], targets[bool_mask])

class MaskedHuberLoss(BaseLoss):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Alignement automatique du masque sur les dimensions du signal [B, C, L]
        if mask.shape != predictions.shape:
            if mask.ndim == 2:
                mask = mask.unsqueeze(1)

            if mask.shape[-1] != predictions.shape[-1]:
                patch_size = predictions.shape[-1] // mask.shape[-1]
                mask = mask.repeat_interleave(patch_size, dim=-1)

            mask = mask.expand_as(predictions)

        bool_mask = mask.type(torch.bool)

        if not bool_mask.any():
            return torch.tensor(0.0, device=targets.device, requires_grad=True)

        return F.huber_loss(predictions[bool_mask], targets[bool_mask], delta=self.delta)