from vivml.SSL.model.head.heads import ProjectionHead, LinearClassificationHead, MLPForecastingHead, CosineDistanceHead, MaskingReconstructionHead, MahalanobisDistanceHead
import torch
import pytest

class TestProjectionHead:

    def test__forward__projects_and_applies_l2_normalization(self):
        # GIVEN
        batch_size = 4
        input_dim = 128
        output_dim = 64

        # pooler output (Batch, Features)
        x = torch.rand(batch_size, input_dim)
        head = ProjectionHead(input_dim=input_dim, hidden_dim=256, output_dim=output_dim)

        # WHEN
        out = head(x)

        # THEN
        assert out.shape == (batch_size, output_dim)

        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)

class TestLinearClassificationHead:

    def test__forward__returns_correct_shape(self):
        # GIVEN
        head = LinearClassificationHead(input_dim=128, num_classes=4)
        x = torch.rand(16, 128)

        # WHEN
        logits = head(x)

        # THEN
        assert logits.shape == (16, 4)
        assert logits.requires_grad == True


class TestMLPForecastingHead:

    def test__forward__returns_correct_output_shape(self):
        # GIVEN
        batch_size = 4
        hidden_size = 128
        output_chunk_length = 20
        in_channels = 3

        head = MLPForecastingHead(
            in_channels=in_channels,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size
        )

        features = torch.rand(batch_size, hidden_size)

        # WHEN
        predictions = head(features)

        # THEN
        assert predictions.shape == (batch_size, in_channels, output_chunk_length)


class TestCosineDistanceHead:
    def test__forward__raises_error_without_fit(self):
        head = CosineDistanceHead()
        x = torch.rand(2, 10)
        with pytest.raises(RuntimeError, match="fit_anchors"):
            head(x)

    def test__fit_anchors_and_forward__calculates_similarity(self):
        head = CosineDistanceHead()

        anchors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        head.fit_anchors(anchors)

        assert head.reference_mean is not None
        assert head.reference_mean.shape == (1, 2)

        test_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        scores = head(test_x)

        assert scores.shape == (3,)
        assert scores[0].item() == pytest.approx(1.0, abs=1e-5)
        assert scores[1].item() == pytest.approx(0.0, abs=1e-5)
        assert scores[2].item() == pytest.approx(-1.0, abs=1e-5)


class TestMahalanobisDistanceHead:
    def test__fit_anchors__calculates_covariance_and_inverse(self):
        head = MahalanobisDistanceHead(epsilon=1e-5)
        anchors = torch.randn(10, 5)

        head.fit_anchors(anchors)

        assert head.reference_mean.shape == (1, 5)
        assert head.reference_cov_inv.shape == (5, 5)

    def test__forward__returns_distances(self):
        head = MahalanobisDistanceHead()
        anchors = torch.randn(20, 8)
        head.fit_anchors(anchors)

        test_x = torch.randn(4, 8)
        distances = head(test_x)

        assert distances.shape == (4,)
        assert torch.all(distances >= 0.0)


class TestMaskingReconstructionHead:

    def test__forward__reshapes_latent_vector_to_original_signal_dimensions(self):
        batch_size = 4
        in_channels = 2
        window_len = 600
        hidden_size = 48
        num_patches = (window_len - 16) // 8 + 1

        head = MaskingReconstructionHead(
            in_channels=in_channels,
            window_len=window_len
        )

        x_latent = torch.rand(batch_size, in_channels, num_patches, hidden_size)

        # WHEN :
        reconstruction = head(x_latent)

        assert reconstruction.shape == (batch_size, in_channels, window_len)

    def test__backward_pass__propagates_gradients_to_reconstructor_weights(self):
        # GIVEN :
        batch_size = 4
        in_channels = 2
        window_len = 100
        hidden_size = 48
        num_patches = (window_len - 16) // 8 + 1

        head = MaskingReconstructionHead(in_channels=in_channels, window_len=window_len)

        x_latent = torch.randn(batch_size, in_channels, num_patches, hidden_size, requires_grad=True)

        # WHEN :
        reconstruction = head(x_latent)

        loss = reconstruction.sum()
        loss.backward()

        # THEN :
        assert head.reconstructor.weight.grad is not None