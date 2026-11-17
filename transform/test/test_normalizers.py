import pytest
import numpy as np
from vivml.SSL.transform.normalizers import MadNormalize, HFNormalize, DriftNormalize

import torch

class TestMadNormalize:

    def test__mad_normalize__centers_and_scales_signal(self):
        # GIVEN :
        batch = torch.tensor([
            [1.0, 3.0, 5.0],
            [10.0, 30.0, 50.0]
        ], dtype=torch.float32)
        normalizer = MadNormalize(epsilon=0.1)

        # WHEN
        normalized_batch = normalizer(batch)

        # THEN
        medians = torch.median(normalized_batch, dim=-1).values
        assert medians[0].item() == 0.0
        assert medians[1].item() == 0.0

    def test__mad_normalize__clamps_on_flat_signal(self):
        # GIVEN :
        flat_batch = torch.zeros((1, 100), dtype=torch.float32)
        epsilon = 0.5
        normalizer = MadNormalize(epsilon=epsilon)

        # WHEN
        normalized_batch = normalizer(flat_batch)

        # THEN
        assert not torch.isnan(normalized_batch).any().item()
        assert torch.all(normalized_batch == 0.0).item()


class TestHFNormalize:

    def test__forward__returns_dynamic_and_static_with_correct_shapes(self):
        # GIVEN
        batch_size = 4
        seq_len = 100
        x = torch.rand(batch_size, seq_len)

        metadata = {
            "stds": {
                "HF": torch.tensor([2.0, 0.5, 1.0, 10.0])
            }
        }

        normalizer = HFNormalize(channel_name="HF", epsilon=1e-6)

        # WHEN
        x_norm, statics = normalizer(x, metadata)

        # THEN
        assert x_norm.shape == (batch_size, 1, seq_len)

        assert statics.shape == (batch_size, 2)

    def test__forward__centers_and_scales_signal_correctly(self):
        # GIVEN
        x = torch.tensor([
            [8.0, 10.0, 12.0]
        ], dtype=torch.float32)

        metadata = {"stds": {"HF": torch.tensor([4.0])}}

        normalizer = HFNormalize(channel_name="HF")

        # WHEN
        x_norm, statics = normalizer(x, metadata)

        # THEN
        expected_x_norm = torch.tensor([[[-0.5, 0.0, 0.5]]])
        assert torch.allclose(x_norm, expected_x_norm)

        expected_log_std = torch.log(torch.tensor(4.0))
        expected_statics = torch.tensor([[2.5, expected_log_std.item()]])
        assert torch.allclose(statics, expected_statics)

    def test__forward__protects_against_zero_std(self):
        # GIVEN
        x = torch.tensor([[5.0, 5.0, 5.0]])  # Signal plat (med = 5.0)

        epsilon = 0.1
        metadata = {"stds": {"HF": torch.tensor([0.0])}}

        normalizer = HFNormalize(channel_name="HF", epsilon=epsilon)

        # WHEN
        x_norm, statics = normalizer(x, metadata)

        # THEN
        assert not torch.isnan(x_norm).any()

        # Le calcul de la statique doit utiliser epsilon au lieu de 0.0
        # med_scaled = 5.0 / 0.1 = 50.0
        # log_std = log(0.1)
        expected_log_std = torch.log(torch.tensor(epsilon))
        expected_statics = torch.tensor([[50.0, expected_log_std.item()]])

        assert torch.allclose(statics, expected_statics)

class TestDriftNormalize:

    def test__forward__returns_dynamic_and_static_with_correct_shapes(self):
        # GIVEN
        batch_size = 2
        seq_len = 50
        x = torch.rand(batch_size, seq_len)

        metadata = {
            "drift": torch.tensor([[0.1, 10.0], [-0.5, 100.0]]),
            "start_idx": torch.tensor([0, 1000]),
            "stds": {"raw": torch.tensor([5.0, 20.0])}
        }

        normalizer = DriftNormalize(channel_name="raw")

        # WHEN
        x_norm, statics = normalizer(x, metadata)

        # THEN
        assert x_norm.shape == (batch_size, 1, seq_len)

        assert statics.shape == (batch_size, 3)

    def test__forward__subtracts_drift_line_and_extracts_statics_correctly(self):
        # GIVEN
        batch_size = 1
        seq_len = 3

        x = torch.tensor([[10.0, 9.0, 8.0]])

        metadata = {
            "drift": torch.tensor([[-1.0, 10.0]]),
            "start_idx": torch.tensor([0]),
            "stds": {"raw": torch.tensor([2.0])}
        }

        normalizer = DriftNormalize(channel_name="raw")

        # WHEN
        x_norm, statics = normalizer(x, metadata)

        # THEN
        expected_x_norm = torch.tensor([[[0.0, 0.0, 0.0]]])
        assert torch.allclose(x_norm, expected_x_norm)

        expected_log_std = torch.log(torch.tensor(2.0))
        expected_statics = torch.tensor([[-1.0, 10.0, expected_log_std.item()]])
        assert torch.allclose(statics, expected_statics)