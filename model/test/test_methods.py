import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from vivml.SSL.model.method.methods import LinearProbingMethod, ForecastingMethod, AnomalyDetectionMethod, AdversarialTiDEMethod, AdversarialSimCLRMethod, MaskedReconstructionMethod
from vivml.SSL.model.method.basemethod import BaseMethod

class DummyMethod(BaseMethod):
    def __init__(self, encoder=None, pooler=None, target_channels=None, normalizers=None):
        super().__init__(target_channels=target_channels, normalizers=normalizers)
        self.encoder = encoder
        self.pooler = pooler

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)


class TestBaseMethod:

    def test__prepare_input_tensor__monochannel_raw_auto_squeezes(self):
        method = DummyMethod()

        batch_size = 4
        seq_len = 100
        batch_x = torch.rand(batch_size, seq_len)

        x_dyn, x_stat = method._prepare_input_tensor(batch_x, stats=None)

        assert x_dyn.shape == (batch_size, 1, seq_len)
        assert torch.equal(x_dyn[:, 0, :], batch_x)
        assert x_stat.shape == (batch_size, 0)

    def test__prepare_input_tensor__multichannel_stacks_correctly(self):
        method = DummyMethod(target_channels=["raw", "HF"])

        batch_size = 4
        seq_len = 100

        batch_x = torch.rand(batch_size, seq_len)
        hf_tensor = torch.rand(batch_size, seq_len)

        stats = {
            "extra_channels": {
                "HF": hf_tensor
            }
        }

        x_dyn, x_stat = method._prepare_input_tensor(batch_x, stats=stats)

        assert x_dyn.shape == (batch_size, 2, seq_len)
        assert torch.equal(x_dyn[:, 0, :], batch_x)
        assert torch.equal(x_dyn[:, 1, :], hf_tensor)
        assert x_stat.shape == (batch_size, 0)

    def test__prepare_input_tensor__applies_normalization_only_to_targeted_channels(self):
        class MockNormalizer(nn.Module):
            def forward(self, x, stats):
                return x * 10.0, torch.tensor([[5.0]] * x.shape[0])

        hf_norm = MockNormalizer()
        normalizers_dict = nn.ModuleDict({"HF": hf_norm})

        method = DummyMethod(target_channels=["raw", "HF"], normalizers=normalizers_dict)

        batch_x = torch.ones(2, 50)
        hf_tensor = torch.ones(2, 50)

        stats = {"extra_channels": {"HF": hf_tensor}}

        x_dyn, x_stat = method._prepare_input_tensor(batch_x, stats=stats)

        assert torch.all(x_dyn[:, 0, :] == 1.0)
        assert torch.all(x_dyn[:, 1, :] == 10.0)
        assert x_stat.shape == (2, 1)
        assert torch.all(x_stat == 5.0)

    def test__get_projection__passes_through_encoder_and_pooler(self):
        mock_encoder = MagicMock()
        mock_encoder.return_value = torch.rand(4, 128)

        mock_pooler = MagicMock()
        mock_pooler.return_value = torch.rand(4, 64)

        method = DummyMethod(encoder=mock_encoder, pooler=mock_pooler, target_channels=["raw"])
        batch_x = torch.rand(4, 100)

        latent = method.get_projection(batch_x, stats=None)

        assert latent.shape == (4, 64)

        args, _ = mock_encoder.call_args
        assert torch.equal(args[0][:, 0, :], batch_x) # x_dyn (squeezed)
        assert args[1].shape == (4, 0) # x_stat (vide)
        mock_pooler.assert_called_once()

    def test__get_projection__works_without_pooler(self):
        mock_encoder = MagicMock()
        mock_encoder.return_value = torch.rand(4, 128)

        method = DummyMethod(encoder=mock_encoder, pooler=None, target_channels=["raw"])
        batch_x = torch.rand(4, 100)

        latent = method.get_projection(batch_x)

        assert latent.shape == (4, 128)
        mock_encoder.assert_called_once()


class TestLinearProbingMethod:

    def test__init__freezes_encoder_when_requested(self):
        mock_encoder = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        mock_head = nn.Linear(10, 2)
        mock_pooler = nn.Identity()
        mock_loss_fn = MagicMock()

        method = LinearProbingMethod(
            encoder=mock_encoder,
            pooler=mock_pooler,
            head=mock_head,
            loss_fn=mock_loss_fn,
            freeze_encoder=True
        )

        for param in method.encoder.parameters():
            assert param.requires_grad == False

        for param in method.head.parameters():
            assert param.requires_grad == True

    def test__forward__calculates_loss_correctly(self):
        mock_encoder = MagicMock()
        mock_encoder.return_value = torch.rand(4, 128)
        mock_pooler = MagicMock()
        mock_pooler.return_value = torch.rand(4, 64)
        mock_head = MagicMock()
        mock_head.return_value = torch.rand(4, 2)
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = torch.tensor(0.5)

        method = LinearProbingMethod(
            encoder=mock_encoder,
            pooler=mock_pooler,
            head=mock_head,
            loss_fn=mock_loss_fn,
            freeze_encoder=True
        )

        x = torch.rand(4, 1, 100)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        loss = method(x, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        mock_encoder.assert_called_once()
        mock_pooler.assert_called_once()
        mock_loss_fn.assert_called_once()


class TestForecastingMethod:

    def test__forward__splits_batch_and_calculates_loss_correctly(self):
        batch_size = 4
        in_channels = 1
        input_chunk_length = 100
        future_chunk_length = 20
        total_length = input_chunk_length + future_chunk_length
        hidden_size = 64

        mock_encoder = MagicMock()
        mock_encoder.return_value = torch.rand(batch_size, hidden_size)
        mock_pooler = MagicMock()
        mock_pooler.return_value = torch.rand(batch_size, hidden_size)
        mock_head = MagicMock()
        mock_head.return_value = torch.rand(batch_size, in_channels, future_chunk_length)
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = torch.tensor(0.25)

        method = ForecastingMethod(
            encoder=mock_encoder,
            pooler=mock_pooler,
            head=mock_head,
            loss_fn=mock_loss_fn,
            input_chunk_length=input_chunk_length,
        )

        batch_x = torch.rand(batch_size, in_channels, total_length)
        loss = method(batch_x)

        assert isinstance(loss, torch.Tensor)

        args_encoder, _ = mock_encoder.call_args
        x_past_received = args_encoder[0]
        assert x_past_received.shape == (batch_size, in_channels, input_chunk_length)

    def test__forward__applies_fft_to_target_and_changes_shape(self):
        batch_size = 2
        in_channels = 1
        input_chunk_length = 50
        future_chunk_length = 20
        total_length = input_chunk_length + future_chunk_length

        expected_fft_length = future_chunk_length // 2 + 1

        mock_encoder = MagicMock()
        mock_encoder.return_value = torch.rand(batch_size, 64)
        mock_pooler = MagicMock()
        mock_pooler.return_value = torch.rand(batch_size, 64)

        mock_head = MagicMock()
        mock_head.return_value = torch.rand(batch_size, 1, expected_fft_length)

        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = torch.tensor(0.5)

        method = ForecastingMethod(
            encoder=mock_encoder,
            pooler=mock_pooler,
            head=mock_head,
            loss_fn=mock_loss_fn,
            input_chunk_length=input_chunk_length,
            target_channels=["raw"]
        )

        batch_x = torch.rand(batch_size, total_length)
        method(batch_x)

        args_loss, _ = mock_loss_fn.call_args
        predictions_received = args_loss[0]
        target_received = args_loss[1]

        assert target_received.shape == (batch_size, 1, expected_fft_length)
        assert predictions_received.shape == (batch_size, 1, expected_fft_length)


class TestAnomalyDetectionMethod:
    def test__init__freezes_encoder(self):
        mock_encoder = nn.Sequential(nn.Linear(10, 10))
        mock_head = MagicMock()

        method = AnomalyDetectionMethod(
            encoder=mock_encoder,
            pooler=None,
            head=mock_head,
            input_chunk_length=10
        )

        for param in method.encoder.parameters():
            assert param.requires_grad == False

    def test__forward__returns_head_scores(self):
        mock_encoder = MagicMock()
        mock_encoder.return_value = torch.rand(4, 128)
        mock_head = MagicMock()
        expected_scores = torch.tensor([0.9, 0.1, 0.8, -0.2])
        mock_head.return_value = expected_scores

        method = AnomalyDetectionMethod(
            encoder=mock_encoder,
            pooler=None,
            head=mock_head,
            input_chunk_length=10
        )

        batch_x = torch.rand(4, 2, 100)
        scores = method(batch_x)

        assert torch.equal(scores, expected_scores)
        mock_head.assert_called_once()

class TestAdversarialTiDEMethod:

    def test__forward__calculates_combined_loss_from_both_heads(self):
        batch_size = 4
        in_channels = 2
        input_chunk_length = 50
        output_chunk_length = 20
        total_length = input_chunk_length + output_chunk_length

        mock_encoder = MagicMock()
        mock_encoder.return_value = torch.rand(batch_size, 128)

        mock_pooler = MagicMock()
        mock_pooler.return_value = torch.rand(batch_size, 64)

        mock_head = MagicMock()
        mock_head.return_value = torch.rand(batch_size, in_channels, output_chunk_length)

        mock_domain_head = MagicMock()
        mock_domain_head.return_value = torch.rand(batch_size, 1000)

        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = torch.tensor(0.5)

        mock_domain_loss_fn = MagicMock()
        mock_domain_loss_fn.return_value = torch.tensor(0.3)

        method = AdversarialTiDEMethod(
            encoder=mock_encoder,
            pooler=mock_pooler,
            head=mock_head,
            domain_head=mock_domain_head,
            loss_fn=mock_loss_fn,
            domain_loss_fn=mock_domain_loss_fn,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            target_dim=in_channels,
            alpha=1.0
        )

        batch_x = torch.rand(batch_size, in_channels, total_length)
        stats = {
            "domain_labels": torch.tensor([0, 500, 250, 999], dtype=torch.long)
        }

        loss = method(batch_x, labels=None, stats=stats)

        assert isinstance(loss, torch.Tensor)
        assert torch.isclose(loss, torch.tensor(1.3))

        mock_encoder.assert_called_once()
        mock_head.assert_called_once()
        mock_domain_head.assert_called_once()
        assert mock_loss_fn.call_count == 2
        mock_domain_loss_fn.assert_called_once()

    def test__forward__raises_keyerror_if_domain_labels_missing(self):
        mock_head = MagicMock()
        mock_head.return_value = torch.rand(4, 2, 20)

        method = AdversarialTiDEMethod(
            encoder=MagicMock(),
            head=mock_head,
            domain_head=MagicMock(),
            loss_fn=MagicMock(),
            domain_loss_fn=MagicMock(),
            input_chunk_length=50,
            output_chunk_length=20,
            target_dim=2
        )

        batch_x = torch.rand(4, 2, 70)
        stats = {"extra_channels": {}}

        with pytest.raises(KeyError):
            method(batch_x, stats=stats)

    def test__backward_pass_triggers_gradient_reversal_and_forecasting(self):
        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2 * 50, 16)

            def forward(self, x_dyn, x_stat=None):
                B = x_dyn.shape[0]
                return self.layer(x_dyn.reshape(B, -1))

        class DummyTiDEHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(16, 2 * 20)

            def forward(self, x):
                B = x.shape[0]
                return self.layer(x).view(B, 2, 20)

        encoder = DummyEncoder()
        head = DummyTiDEHead()
        domain_head = nn.Linear(16, 1000)

        method = AdversarialTiDEMethod(
            encoder=encoder,
            head=head,
            domain_head=domain_head,
            loss_fn=nn.MSELoss(),
            domain_loss_fn=nn.CrossEntropyLoss(),
            input_chunk_length=50,
            output_chunk_length=20,
            target_dim=2,
            alpha=0.5
        )

        batch_x = torch.rand(4, 2, 70)
        stats = {"domain_labels": torch.tensor([0, 500, 250, 999], dtype=torch.long)}

        loss = method(batch_x, stats=stats)
        loss.backward()

        for param in encoder.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


class TestAdversarialSimCLRMethod:

    def test__forward__calculates_combined_loss_with_batch_concatenation(self):
        # GIVEN :
        batch_size = 4
        in_channels = 2
        window_len = 600
        doubled_batch = batch_size * 2

        mock_encoder = MagicMock()
        mock_encoder.return_value = torch.rand(doubled_batch, 128)

        mock_pooler = MagicMock()
        mock_pooler.return_value = torch.rand(doubled_batch, 64)

        mock_head = MagicMock()
        mock_head.return_value = torch.rand(doubled_batch, 32)

        mock_domain_head = MagicMock()
        mock_domain_head.return_value = torch.rand(doubled_batch, 1000)

        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = torch.tensor(1.5)

        mock_domain_loss_fn = MagicMock()
        mock_domain_loss_fn.return_value = torch.tensor(0.5)

        mock_augmenter = MagicMock(side_effect=lambda x: x.clone())
        mock_static_augmenter = MagicMock(side_effect=lambda x: x.clone())

        method = AdversarialSimCLRMethod(
            encoder=mock_encoder,
            pooler=mock_pooler,
            head=mock_head,
            domain_head=mock_domain_head,
            augmenter=mock_augmenter,
            static_augmenter=mock_static_augmenter,
            loss_fn=mock_loss_fn,
            domain_loss_fn=mock_domain_loss_fn,
            alpha=1.0
        )

        method._prepare_input_tensor = MagicMock(
            return_value=(torch.rand(batch_size, in_channels, window_len), torch.rand(batch_size, 5))
        )

        batch_x = torch.rand(batch_size, in_channels, window_len)
        stats = {
            "domain_labels": torch.tensor([0, 500, 250, 999], dtype=torch.long)
        }

        # WHEN :
        loss = method(batch_x, labels=None, stats=stats)

        # THEN :
        assert isinstance(loss, torch.Tensor)
        assert torch.isclose(loss, torch.tensor(2.0))

        mock_encoder.assert_called_once()
        mock_head.assert_called_once()
        mock_domain_head.assert_called_once()
        mock_loss_fn.assert_called_once()
        mock_domain_loss_fn.assert_called_once()

    def test__forward__raises_keyerror_if_domain_labels_missing(self):
        # GIVEN :
        method = AdversarialSimCLRMethod(
            encoder=MagicMock(),
            head=MagicMock(),
            domain_head=MagicMock(),
            augmenter=MagicMock(side_effect=lambda x: x),
            loss_fn=MagicMock(),
            domain_loss_fn=MagicMock()
        )
        method._prepare_input_tensor = MagicMock(
            return_value=(torch.rand(4, 2, 600), torch.tensor([]))
        )

        batch_x = torch.rand(4, 2, 600)
        stats = {"extra_channels": {}}

        # WHEN / THEN :
        with pytest.raises(KeyError):
            method(batch_x, stats=stats)

    def test__backward_pass_triggers_gradient_reversal_and_simclr(self):
        # GIVEN :
        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2 * 600, 16)

            def forward(self, x_dyn, x_stat=None):
                B = x_dyn.shape[0]
                return self.layer(x_dyn.reshape(B, -1))

        class DummyProjectionHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(16, 8)

            def forward(self, x):
                return self.layer(x)

        encoder = DummyEncoder()
        head = DummyProjectionHead()
        domain_head = nn.Linear(16, 1000)

        method = AdversarialSimCLRMethod(
            encoder=encoder,
            head=head,
            domain_head=domain_head,
            augmenter=MagicMock(side_effect=lambda x: x.clone()),
            loss_fn=lambda z: z.sum(),  # Perte factice
            domain_loss_fn=nn.CrossEntropyLoss(),
            alpha=0.5
        )
        method._prepare_input_tensor = MagicMock(
            return_value=(torch.rand(4, 2, 600), torch.tensor([]))
        )

        batch_x = torch.rand(4, 2, 600)
        stats = {"domain_labels": torch.tensor([0, 500, 250, 999], dtype=torch.long)}

        # WHEN :
        loss = method(batch_x, stats=stats)
        loss.backward()

        # THEN :
        for param in encoder.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)



class TestMaskedReconstructionMethod:

    def test__forward__reconstructs_and_compares_to_uncorrupted_input(self):
        # GIVEN :
        batch_size = 4
        in_channels = 2
        seq_len = 100

        mock_encoder = MagicMock(return_value=torch.rand(batch_size, 128))
        mock_pooler = MagicMock(return_value=torch.rand(batch_size, 64))
        mock_head = MagicMock(return_value=torch.rand(batch_size, in_channels, seq_len))

        mock_corrupted_signal = torch.zeros(batch_size, in_channels, seq_len)
        mock_augmenter = MagicMock(return_value=mock_corrupted_signal)

        mock_loss_fn = MagicMock(return_value=torch.tensor(0.5))

        method = MaskedReconstructionMethod(
            encoder=mock_encoder,
            pooler=mock_pooler,
            head=mock_head,
            augmenter=mock_augmenter,
            loss_fn=mock_loss_fn
        )

        original_signal = torch.rand(batch_size, in_channels, seq_len)
        method._prepare_input_tensor = MagicMock(
            return_value=(original_signal, torch.tensor([]))
        )

        # WHEN :
        loss = method(original_signal, labels=None, stats=None)

        # THEN :
        assert isinstance(loss, torch.Tensor)

        args_encoder, _ = mock_encoder.call_args
        assert torch.equal(args_encoder[0], mock_corrupted_signal)

        args_loss, _ = mock_loss_fn.call_args
        reconstruction_received, target_received = args_loss
        assert torch.equal(target_received, original_signal)

    def test__forward__works_without_augmenter_fallback(self):
        method = MaskedReconstructionMethod(
            encoder=MagicMock(return_value=torch.rand(2, 128)),
            pooler=None,
            head=MagicMock(return_value=torch.rand(2, 1, 50)),
            augmenter=None,
            loss_fn=MagicMock(return_value=torch.tensor(0.1))
        )

        batch_x = torch.rand(2, 1, 50)
        method._prepare_input_tensor = MagicMock(return_value=(batch_x, torch.tensor([])))

        # WHEN :
        loss = method(batch_x)

        # THEN :
        assert isinstance(loss, torch.Tensor)
        method.encoder.assert_called_once()