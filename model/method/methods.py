import torch
import torch.nn as nn
import math
from .basemethod import BaseMethod, GradientReversal
from vivml.SSL.evaluation.metrics import ClusteringMetrics
import logging

logger = logging.getLogger(__name__)



class SimCLRMethod(BaseMethod):
    def __init__(self, encoder: nn.Module, pooler: nn.Module, head: nn.Module,
                 augmenter: nn.Module, loss_fn: nn.Module,
                 static_augmenter: nn.Module = None,
                 target_channels: list[str] = None, normalizers: nn.ModuleDict = None):
        super().__init__(target_channels, normalizers = nn.ModuleDict())
        self.encoder = encoder
        self.pooler = pooler
        self.head = head
        self.augmenter = augmenter
        self.static_augmenter = static_augmenter
        self.loss_fn = loss_fn

    def forward(self, batch_x: torch.Tensor, labels: torch.Tensor = None, stats: dict = None, *args) -> torch.Tensor:
        x_dyn, x_stat = self._prepare_input_tensor(batch_x, stats)

        x1 = self.augmenter(x_dyn)
        x2 = self.augmenter(x_dyn)
        x_concat = torch.cat([x1, x2], dim=0)

        if x_stat.numel() > 0:
            if self.static_augmenter is not None:
                x_stat1 = self.static_augmenter(x_stat)
                x_stat2 = self.static_augmenter(x_stat)
            else:
                x_stat1, x_stat2 = x_stat, x_stat

            x_stat_concat = torch.cat([x_stat1, x_stat2], dim=0)
        else:
            x_stat_concat = x_stat

        latent = self.encoder(x_concat, x_stat_concat)

        if self.pooler is not None:
            latent = self.pooler(latent)

        z = self.head(latent)
        return self.loss_fn(z)

    def compute_training_metrics(self, batch_x: torch.Tensor, stats: dict = None) -> dict:

        x_dyn, x_stat = self._prepare_input_tensor(batch_x, stats)

        limit = min(25, x_dyn.size(0))
        x_sub = x_dyn[:limit]
        x_stat_sub = x_stat[:limit] if x_stat.numel() > 0 else x_stat

        x1 = self.augmenter(x_sub)
        x2 = self.augmenter(x_sub)

        self.eval()
        with torch.no_grad():
            latent1 = self.encoder(x1, x_stat_sub)
            latent2 = self.encoder(x2, x_stat_sub)
            latent_all = self.encoder(x_sub, x_stat_sub)

            if self.pooler is not None:
                latent1 = self.pooler(latent1)
                latent2 = self.pooler(latent2)
                latent_all = self.pooler(latent_all)

            z1 = self.head(latent1)
            z2 = self.head(latent2)
            z_all = self.head(latent_all)
        self.train()

        return ClusteringMetrics.compute_geometry(z1, z2, z_all)


class LinearProbingMethod(BaseMethod):
    def __init__(self, encoder: nn.Module, pooler: nn.Module, head: nn.Module, loss_fn: nn.Module,
                 freeze_encoder: bool = True, input_chunk_length: int = None,
                 target_channels: list[str] = None, normalizers: nn.ModuleDict = None):
        super().__init__(target_channels, normalizers)
        self.encoder = encoder
        self.pooler = pooler
        self.head = head
        self.loss_fn = loss_fn
        self.input_chunk_length = input_chunk_length

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, batch_x: torch.Tensor, labels: torch.Tensor, stats: dict = None, *args) -> torch.Tensor:

        latent = self.get_projection(batch_x, stats)
        logits = self.head(latent)

        if labels.dim() > 1:
            labels = labels.squeeze()

        return self.loss_fn(logits, labels.long())

    def get_prediction(self, x: torch.Tensor, stats: dict = None) -> torch.Tensor:
        latent = self.get_projection(x, stats)
        logits = self.head(latent)
        return torch.softmax(logits, dim=1)

class TiDEMethod(BaseMethod):
    def __init__(self, encoder: nn.Module, head: nn.Module, loss_fn: nn.Module,
                 input_chunk_length: int, output_chunk_length: int, pooler: nn.Module = None,
                 target_channels: list[str] = None, normalizers: nn.ModuleDict = None, target_dim: int = 2):
        super().__init__(target_channels, normalizers)
        self.encoder = encoder
        self.pooler = pooler
        self.head = head
        self.loss_fn = loss_fn
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.target_dim = target_dim

        self.global_residual = nn.Linear(self.input_chunk_length, self.output_chunk_length)

    def forward(self, batch_x: torch.Tensor, labels: torch.Tensor = None, stats: dict = None, *args) -> torch.Tensor:
        x_dyn, x_stat = self._prepare_input_tensor(batch_x, stats)

        x_past = x_dyn[:, :, :self.input_chunk_length]
        y_future = x_dyn[:, :self.target_dim, self.input_chunk_length:]
        target_past = x_past[:, :self.target_dim, :]

        latent = self.encoder(x_past, x_stat)

        if self.pooler is not None:
            latent = self.pooler(latent)

        predictions = self.head(latent)
        final_predictions = predictions + self.global_residual(target_past)

        loss_hf = self.loss_fn(
            torch.abs(torch.fft.rfft(final_predictions[:, 0, :], dim=-1)),
            torch.abs(torch.fft.rfft(y_future[:, 0, :], dim=-1))
        )
        loss_drift = self.loss_fn(final_predictions[:, 1, :], y_future[:, 1, :])

        return loss_hf + loss_drift


class ForecastingMethod(BaseMethod):
    def __init__(self, encoder: nn.Module, head: nn.Module, loss_fn: nn.Module,
                 input_chunk_length: int, pooler: nn.Module, target_channels: list[str] = None,
                 normalizers: nn.ModuleDict = None):
        super().__init__(target_channels, normalizers)
        self.encoder = encoder
        self.pooler = pooler
        self.head = head
        self.loss_fn = loss_fn
        self.input_chunk_length = input_chunk_length

    def forward(self, batch_x: torch.Tensor, labels: torch.Tensor = None, stats: dict = None, *args) -> torch.Tensor:
        x_dyn, x_stat = self._prepare_input_tensor(batch_x, stats)

        x_past = x_dyn[:, :, :self.input_chunk_length]
        y_future = x_dyn[:, 0:1, self.input_chunk_length:]

        fft_future = torch.fft.rfft(y_future, dim=-1)
        y_future = torch.abs(fft_future)

        latent = self.encoder(x_past, x_stat)
        if self.pooler is not None:
            latent = self.pooler(latent)

        predictions = self.head(latent)

        if predictions.ndim == 3 and y_future.ndim == 2:
            y_future = y_future.unsqueeze(1)

        return self.loss_fn(predictions, y_future)

    def compute_training_metrics(self, batch_x: torch.Tensor, stats: dict = None) -> dict:
        with torch.no_grad():
            latent = self.get_projection(batch_x, stats)
        return {
            "latent_mean": latent.mean().item(),
            "latent_std": latent.std().item()
        }


class AnomalyDetectionMethod(BaseMethod):
    def __init__(self, encoder: nn.Module, pooler: nn.Module, head: nn.Module,
                 input_chunk_length: int, target_channels: list[str] = None, normalizers: nn.ModuleDict = None):
        super().__init__(target_channels, normalizers)
        self.encoder = encoder
        self.pooler = pooler
        self.head = head
        self.input_chunk_length = input_chunk_length

        for param in self.encoder.parameters():
            param.requires_grad = False

    def fit_anchors(self, batch_x_healthy: torch.Tensor, stats: dict = None):
        self.eval()
        with torch.no_grad():
            latent = self.get_projection(batch_x_healthy, stats)

        if hasattr(self.head, "fit_anchors"):
            self.head.fit_anchors(latent)
        else:
            raise ValueError("fit_anchors() not implemented for this head.")

    def forward(self, batch_x: torch.Tensor, stats: dict = None, *args) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            latent = self.get_projection(batch_x, stats)
        return self.head(latent)


class AdversarialTiDEMethod(BaseMethod):
    def __init__(
            self,
            encoder: nn.Module,
            head: nn.Module,
            domain_head: nn.Module,
            loss_fn: nn.Module,
            domain_loss_fn: nn.Module,
            input_chunk_length: int,
            output_chunk_length: int,
            alpha: float = 1.0,
            pooler: nn.Module = None,
            target_channels: list[str] = None,
            normalizers: nn.ModuleDict = None,
            target_dim: int = 2
    ):
        super().__init__(target_channels, normalizers)
        self.encoder = encoder
        self.pooler = pooler
        self.head = head
        self.domain_head = domain_head
        self.loss_fn = loss_fn
        self.domain_loss_fn = domain_loss_fn
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.target_dim = target_dim
        self.global_residual = nn.Linear(self.input_chunk_length, self.output_chunk_length)
        self.max_alpha = alpha
        self.alpha = 0.0

    def on_epoch_start(self, current_epoch: int, total_epochs: int):
        p = current_epoch / max(1, total_epochs - 1)
        self.alpha = self.max_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


    def forward(self, batch_x: torch.Tensor, labels: torch.Tensor = None, stats: dict = None, *args) -> torch.Tensor:
        x_dyn, x_stat = self._prepare_input_tensor(batch_x, stats)

        x_past = x_dyn[:, :, :self.input_chunk_length]
        y_future = x_dyn[:, :self.target_dim, self.input_chunk_length:]
        target_past = x_past[:, :self.target_dim, :]

        latent = self.encoder(x_past, x_stat)
        if self.pooler is not None:
            latent = self.pooler(latent)

        predictions = self.head(latent)
        final_predictions = predictions + self.global_residual(target_past)

        loss_hf = self.loss_fn(
            torch.abs(torch.fft.rfft(final_predictions[:, 0, :], dim=-1)),
            torch.abs(torch.fft.rfft(y_future[:, 0, :], dim=-1))
        )
        loss_drift = self.loss_fn(final_predictions[:, 1, :], y_future[:, 1, :])

        loss_forecasting = loss_hf + loss_drift

        latent_reversed = GradientReversal.apply(latent, self.alpha)
        domain_preds = self.domain_head(latent_reversed)

        domain_labels = stats["domain_labels"].to(latent.device)
        if domain_labels.dim() > 1:
            domain_labels = domain_labels.squeeze()

        loss_domain = self.domain_loss_fn(domain_preds, domain_labels.long())

        return loss_forecasting + loss_domain


class AdversarialSimCLRMethod(BaseMethod):
    def __init__(self, encoder, head, domain_head, augmenter, loss_fn,
                 domain_loss_fn,  pooler: nn.Module = None, alpha=0.1, static_augmenter=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.pooler = pooler
        self.head = head
        self.domain_head = domain_head
        self.augmenter = augmenter
        self.static_augmenter = static_augmenter
        self.loss_fn = loss_fn
        self.domain_loss_fn = domain_loss_fn
        self.max_alpha = alpha
        self.alpha = 0.0

    def on_epoch_start(self, current_epoch, total_epochs):
        p = current_epoch / max(1, total_epochs - 1)
        self.alpha = self.max_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

    def forward(self, batch_x, labels=None, stats=None, *args):
        x_dyn, x_stat = self._prepare_input_tensor(batch_x, stats)

        x1, x2 = self.augmenter(x_dyn), self.augmenter(x_dyn)
        x_concat = torch.cat([x1, x2], dim=0)

        if self.static_augmenter and x_stat.numel() > 0:
            x_stat_concat = torch.cat([self.static_augmenter(x_stat), self.static_augmenter(x_stat)], dim=0)
        else:
            x_stat_concat = torch.cat([x_stat, x_stat], dim=0)

        latent = self.encoder(x_concat, x_stat_concat)
        if self.pooler: latent = self.pooler(latent)
        z = self.head(latent)
        loss_simclr = self.loss_fn(z)

        latent_reversed = GradientReversal.apply(latent, self.alpha)
        domain_preds = self.domain_head(latent_reversed)

        domain_labels = stats["domain_labels"].to(latent.device)
        if domain_labels.dim() > 1:
            domain_labels = domain_labels.squeeze()

        domain_labels_concat = torch.cat([domain_labels, domain_labels], dim=0)
        loss_domain = self.domain_loss_fn(domain_preds, domain_labels_concat.long())

        return loss_simclr + loss_domain

class MaskedReconstructionMethod(BaseMethod):
    def __init__(self, encoder: nn.Module, head: nn.Module, loss_fn: nn.Module,
                 augmenter: nn.Module = None, pooler: nn.Module = None,
                 target_channels: list[str] = None, normalizers: nn.ModuleDict = None):
        super().__init__(target_channels, normalizers)
        self.encoder = encoder
        self.pooler = pooler
        self.head = head
        self.augmenter = augmenter
        self.loss_fn = loss_fn

    def forward(self, batch_x: torch.Tensor, labels: torch.Tensor = None, stats: dict = None, *args) -> torch.Tensor:

        x_dyn, x_stat = self._prepare_input_tensor(batch_x, stats)

        if self.augmenter is not None:
            x_corrupted = self.augmenter(x_dyn)
        else:
            x_corrupted = x_dyn

        latent = self.encoder(x_corrupted, x_stat)

        if self.pooler is not None:
            latent = self.pooler(latent)

        reconstruction = self.head(latent)

        return self.loss_fn(reconstruction, x_dyn)

class TSPulseSSLMethod(BaseMethod):
    def __init__(self, encoder: nn.Module, head: nn.Module, loss_fn: nn.Module,
                 target_channels: list[str] = None, normalizers: nn.ModuleDict = None):
        super().__init__(target_channels, normalizers)
        self.encoder = encoder
        self.head = head
        self.loss_fn = loss_fn

    def forward(self, batch_x: torch.Tensor, labels: torch.Tensor = None, stats: dict = None, *args) -> torch.Tensor:
        x_dyn, _ = self._prepare_input_tensor(batch_x, stats)



        encoder_out = self.encoder(past_values=x_dyn, enable_masking=True)

        head_out = self.head(
            decoder_input=encoder_out.last_hidden_state,
            loc=encoder_out.loc,
            scale=encoder_out.scale
        )

        x_dyn_transposed = x_dyn.transpose(1, 2)

        if head_out.reconstruction_outputs.shape != x_dyn_transposed.shape:
            logger.error(
                f"Shape mismatch! Preds: {head_out.reconstruction_outputs.shape}, Targets: {x_dyn_transposed.shape}")
            raise ValueError("Broadcasting trap detected in SSL Loss.")

        return self.loss_fn(
            predictions=head_out.reconstruction_outputs,
            targets=x_dyn_transposed,
            mask=encoder_out.mask
        )

    def get_prediction(self, batch_x: torch.Tensor, stats: dict = None) -> torch.Tensor:
        x_dyn, _ = self._prepare_input_tensor(batch_x, stats)

        encoder_out = self.encoder(past_values=x_dyn, enable_masking=False)

        head_out = self.head(
            decoder_input=encoder_out.last_hidden_state,
            loc=encoder_out.loc,
            scale=encoder_out.scale
        )

        return head_out.reconstruction_outputs

    def get_projection(self, batch_x: torch.Tensor, stats: dict = None) -> torch.Tensor:
        x_dyn, _ = self._prepare_input_tensor(batch_x, stats)
        out = self.encoder(past_values=x_dyn, enable_masking=False)

        latent = out.last_hidden_state if hasattr(out, "last_hidden_state") else out

        if latent.ndim == 4:
            return latent.max(dim=1)[0].max(dim=1)[0]

        if latent.ndim == 3:
            return latent.max(dim=1)[0]

        raise ValueError(f"shape error before pooling: {latent.shape}")


class MomentSSLMethod(BaseMethod):
    def __init__(self, encoder: nn.Module, loss_fn: nn.Module,
                 target_channels: list[str] = None, normalizers: nn.ModuleDict = None):
        super().__init__(target_channels, normalizers)
        self.encoder = encoder
        self.loss_fn = loss_fn

    def forward(self, batch_x: torch.Tensor, labels: torch.Tensor = None, stats: dict = None, *args) -> torch.Tensor:
        x_dyn, _ = self._prepare_input_tensor(batch_x, stats)

        outputs = self.encoder(x_dynamic=x_dyn, enable_masking=True)

        mask = getattr(outputs, "pre_training_mask", None)
        if mask is None:
            mask = getattr(outputs, "input_mask", getattr(outputs, "mask", None))

        if mask is None:
            raise AttributeError(
                f"Impossible de localiser le masque dans la sortie MOMENT. Attributs disponibles: {[a for a in dir(outputs) if not a.startswith('_')]}"
            )

        return self.loss_fn(
            predictions=outputs.reconstruction,
            targets=x_dyn,
            mask=mask
        )

    def get_prediction(self, batch_x: torch.Tensor, stats: dict = None) -> torch.Tensor:
        x_dyn, _ = self._prepare_input_tensor(batch_x, stats)
        outputs = self.encoder(x_dynamic=x_dyn, enable_masking=True)
        return outputs.reconstruction

    def get_projection(self, batch_x: torch.Tensor, stats: dict = None) -> torch.Tensor:
        x_dyn, _ = self._prepare_input_tensor(batch_x, stats)
        latent = self.encoder(x_dynamic=x_dyn)

        logger.info(f"get_projection: latent.ndim = {latent.ndim}, latent.size = {list(latent.size())}")

        if latent.ndim == 4:
            return latent.max(dim=1)[0].max(dim=1)[0]

        if latent.ndim == 3:
            return latent.max(dim=1)[0]

        raise ValueError(f"shape error before pooling: {latent.shape}")