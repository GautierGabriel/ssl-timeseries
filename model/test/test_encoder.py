import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from vivml.SSL.model.encoder.encoders import PatchTSTEncoder, MomentEncoder, TSPulseEncoder, MLPEncoder, ResidualBlock, Conv1DEncoder, Conv1DEncoderLarge


class TestConv1DEncoder:

    def test__output_dim__returns_correct_instantiated_value(self):
        # GIVEN
        expected_dim = 256

        # WHEN
        encoder = Conv1DEncoder(output_dim=expected_dim)

        # THEN
        assert encoder.output_dim == expected_dim

    def test__forward__processes_tensor_and_reduces_dimensions(self):
        # GIVEN
        x = torch.rand(4, 1, 600)
        encoder = Conv1DEncoder(in_channels=1, hidden_channels=64, output_dim=128)

        # WHEN
        out = encoder(x)

        # THEN
        assert out.shape[0] == 4
        assert out.shape[1] == 128
        assert out.shape[2] == 38

    def test__forward__supports_gpu_device_execution(self):
        # GIVEN
        if not torch.cuda.is_available():
            pytest.skip("CUDA device unavailable.")

        device = torch.device("cuda")
        x = torch.rand(2, 1, 600, device=device)
        encoder = Conv1DEncoder().to(device)

        # WHEN
        out = encoder(x)

        # THEN
        assert out.device.type == "cuda"


class TestConv1DEncoderLarge:

    def test__output_dim__returns_correct_instantiated_value(self):
        # GIVEN
        expected_dim = 128

        # WHEN
        encoder = Conv1DEncoderLarge(output_dim=expected_dim)

        # THEN
        assert encoder.output_dim == expected_dim

    def test__forward__processes_tensor_and_returns_correct_shape(self):
        # GIVEN
        batch_size = 4
        in_channels = 1
        seq_len = 600
        hidden_channels = 64
        output_dim = 128
        num_layers = 4

        x = torch.rand(batch_size, in_channels, seq_len)
        encoder = Conv1DEncoderLarge(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            output_dim=output_dim,
            num_layers=num_layers
        )

        # WHEN
        out = encoder(x_dynamic=x)

        # THEN
        assert out.dim() == 3
        assert out.shape[0] == batch_size
        assert out.shape[1] == output_dim
        assert out.shape[2] > 0

    def test__forward__supports_gpu_device_execution(self):
        # GIVEN
        if not torch.cuda.is_available():
            pytest.skip("CUDA device unavailable.")

        device = torch.device("cuda")
        x = torch.rand(2, 1, 600, device=device)
        encoder = Conv1DEncoderLarge().to(device)

        # WHEN
        out = encoder(x_dynamic=x)

        # THEN
        assert out.device.type == "cuda"

    def test__init__supports_deep_architectures(self):
        # GIVEN
        num_layers = 6
        hidden_channels = 128

        # WHEN
        encoder = Conv1DEncoderLarge(
            hidden_channels=hidden_channels,
            output_dim=256,
            num_layers=num_layers
        )

        # THEN
        assert isinstance(encoder.network, nn.Sequential)
        assert encoder.output_dim == 256

class TestMLPEncoder:

    def test__init__sets_properties_correctly(self):
        # GIVEN
        hidden_size = 128
        in_features = 64

        # WHEN
        encoder = MLPEncoder(in_features=in_features, hidden_size=hidden_size)

        # THEN
        assert encoder.output_dim == hidden_size

    def test__forward__returns_correct_shape(self):
        # GIVEN
        batch_size = 4
        in_features = 100
        hidden_size = 64

        encoder = MLPEncoder(
            in_features=in_features,
            hidden_size=hidden_size
        )

        x = torch.rand(batch_size, in_features)

        # WHEN
        features = encoder(x)

        # THEN
        assert features.shape == (batch_size, hidden_size)


class TestResidualBlock:

    def test__forward__returns_correct_shape(self):
        # GIVEN
        batch_size = 16
        input_dim = 10
        output_dim = 5
        hidden_size = 32

        block = ResidualBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            dropout=0.1,
            use_layer_norm=True
        )
        x = torch.rand(batch_size, input_dim)

        # WHEN
        out = block(x)

        # THEN
        assert out.shape == (batch_size, output_dim)

    def test__init__creates_layer_norm_when_requested(self):
        # GIVEN
        output_dim = 5

        # WHEN
        block = ResidualBlock(
            input_dim=10,
            output_dim=output_dim,
            hidden_size=32,
            dropout=0.1,
            use_layer_norm=True
        )

        # THEN
        assert block.layer_norm is not None
        assert isinstance(block.layer_norm, nn.LayerNorm)
        assert block.layer_norm.normalized_shape[0] == output_dim

    def test__init__skips_layer_norm_when_false(self):
        # GIVEN

        # WHEN
        block = ResidualBlock(
            input_dim=10,
            output_dim=5,
            hidden_size=32,
            dropout=0.1,
            use_layer_norm=False
        )

        # THEN
        assert block.layer_norm is None

    def test__forward__works_without_layer_norm(self):
        # GIVEN
        batch_size = 8
        input_dim = 10
        output_dim = 5

        block = ResidualBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=32,
            dropout=0.1,
            use_layer_norm=False
        )
        x = torch.rand(batch_size, input_dim)

        # WHEN
        out = block(x)

        # THEN
        assert out.shape == (batch_size, output_dim)

    def test__init__sets_dropout_probability_correctly(self):
        # GIVEN
        expected_dropout_p = 0.3

        # WHEN
        block = ResidualBlock(
            input_dim=10,
            output_dim=5,
            hidden_size=32,
            dropout=expected_dropout_p,
            use_layer_norm=True
        )

        # THEN
        dropout_layer = next(module for module in block.dense if isinstance(module, nn.Dropout))
        assert dropout_layer.p == expected_dropout_p

class TestPatchTSTEncoder:

    @patch("vivml.SSL.model.encoder.encoders.PatchTSTModel")
    def test__init__sets_properties_and_dimensions(self, mock_patchtst):
        # GIVEN
        mock_patchtst.return_value = MagicMock()

        # WHEN
        encoder = PatchTSTEncoder(in_channels=2, window_len=512, hidden_size=128)

        # THEN
        assert encoder.output_dim == 128
        assert encoder.in_channels == 2

    @patch("vivml.SSL.model.encoder.encoders.PatchTSTModel")
    def test__forward__processes_tensor_and_returns_mean_embedding(self, mock_patchtst):
        # GIVEN
        batch_size, in_channels, seq_len = 4, 2, 512
        hidden_size = 128
        num_patches = 32

        expected_shape = (batch_size, in_channels, num_patches, hidden_size)

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.rand(*expected_shape)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_outputs
        mock_patchtst.return_value = mock_model_instance

        encoder = PatchTSTEncoder(in_channels=in_channels, window_len=seq_len, hidden_size=hidden_size)
        x = torch.rand(batch_size, in_channels, seq_len)

        # WHEN
        out = encoder(x)

        # THEN
        assert out.shape == expected_shape

class TestMomentEncoder:

    @patch("vivml.SSL.model.encoder.encoders.MOMENTPipeline.from_pretrained")
    def test__forward__processes_tensor_and_returns_embeddings(self, mock_moment):
        # GIVEN
        batch_size, in_channels, seq_len = 4, 1, 512

        mock_pipeline = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.embeddings = torch.rand(batch_size, 1024)
        mock_pipeline.embed.return_value = mock_outputs

        mock_moment.return_value = mock_pipeline

        encoder = MomentEncoder(in_channels=in_channels, freeze_backbone=True)
        x = torch.rand(batch_size, in_channels, seq_len)

        # WHEN
        out = encoder(x)

        # THEN
        assert out.shape == (batch_size, 1024)


class TestTSPulseEncoder:


    @patch("vivml.SSL.model.encoder.encoders.TSPulseForReconstruction.from_pretrained")
    def test__forward__processes_tensor_and_reduces_dimensions(self, mock_tspulse):
        # GIVEN
        batch_size, in_channels, seq_len = 4, 2, 512
        hidden_size = 48  # TSPulse defaut
        num_patches = 69

        expected_shape = (batch_size, in_channels, num_patches, hidden_size)

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.rand(*expected_shape)
        mock_tspulse.return_value.backbone.return_value = mock_outputs

        encoder = TSPulseEncoder(in_channels=in_channels)
        x = torch.rand(batch_size, in_channels, seq_len)

        # WHEN
        out = encoder(x)

        # THEN
        assert out.shape == expected_shape

