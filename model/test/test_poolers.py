import pytest
import torch
from vivml.SSL.model.pooler.poolers import IdentityPooler, AvgPooler, MaxPooler


class TestPoolers:

    @pytest.fixture
    def sample_encoder_output(self):
        x = torch.zeros(2, 128, 38)
        x[:, :, 0] = 10.0
        x[:, :, -1] = 2.0
        return x

    def test__identity_pooler__preserves_exact_shapes_and_values(self, sample_encoder_output):
        # GIVEN
        pooler = IdentityPooler()

        # WHEN
        out = pooler(sample_encoder_output)

        # THEN
        assert out.shape == sample_encoder_output.shape
        assert torch.equal(out, sample_encoder_output)

    def test__avg_pooler__reduces_time_dimension_via_mean(self, sample_encoder_output):
        # GIVEN
        pooler = AvgPooler()

        # WHEN
        out = pooler(sample_encoder_output)

        # THEN
        assert out.shape == (2, 128)
        expected_mean = 12.0 / 38.0
        assert torch.allclose(out, torch.tensor(expected_mean), atol=1e-4)

    def test__max_pooler__reduces_time_dimension_via_maximum(self, sample_encoder_output):
        # GIVEN
        pooler = MaxPooler()

        # WHEN
        out = pooler(sample_encoder_output)

        # THEN
        assert out.shape == (2, 128)
        assert torch.all(out == 10.0)