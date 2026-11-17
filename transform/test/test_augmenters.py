import pytest
import torch
from vivml.SSL.transform.augmenters import TimeSeriesAugmenter, StaticAugmenter, PatchMaskingAugmenter

class TestTimeSeriesAugmenter:

    @pytest.fixture
    def sample_batch(self):
        # Shape: (Batch_size=4, Channels=1, Time_steps=600)
        return torch.rand(4, 1, 600)

    def test__augmenter__sequential_calls_generate_distinct_views(self, sample_batch):
        # GIVEN :
        augmenter = TimeSeriesAugmenter(p_noise=1.0, p_scale=1.0, p_shift=1.0, p_mask=1.0)

        # WHEN :
        x1 = augmenter(sample_batch)
        x2 = augmenter(sample_batch)

        # THEN
        assert x1.shape == sample_batch.shape
        assert x2.shape == sample_batch.shape
        assert not torch.equal(x1, x2)

    def test__time_mask__sets_exact_number_of_zeros(self, sample_batch):
        # GIVEN :
        mask_ratio = 0.1
        augmenter = TimeSeriesAugmenter(mask_ratio=mask_ratio)

        # WHEN :
        masked_batch = augmenter.time_mask(sample_batch)

        # THEN :
        for i in range(sample_batch.size(0)):
            num_zeros = (masked_batch[i] == 0.0).sum().item()
            assert num_zeros == 60

    def test__time_shift__preserves_all_original_values_via_rotation(self, sample_batch):
        # GIVEN :
        augmenter = TimeSeriesAugmenter(max_shift=5)

        # WHEN
        shifted_batch = augmenter.time_shift(sample_batch)

        # THEN :
        for i in range(sample_batch.size(0)):
            assert torch.allclose(shifted_batch[i].sum(), sample_batch[i].sum(), atol=1e-4)

class TestStaticAugmenter:

    @pytest.fixture
    def sample_stats(self):
        return torch.ones(4, 5)

    def test__call__sequential_calls_generate_distinct_views(self, sample_stats):
        # GIVEN :
        augmenter = StaticAugmenter(noise_std=0.05, p_dropout=0.2)

        # WHEN :
        x1 = augmenter(sample_stats)
        x2 = augmenter(sample_stats)

        # THEN :
        assert x1.shape == sample_stats.shape
        assert x2.shape == sample_stats.shape
        assert not torch.equal(x1, x2)

    def test__call__handles_empty_tensors(self):
        # GIVEN :
        augmenter = StaticAugmenter()
        empty_stat = torch.tensor([])

        # WHEN :
        out = augmenter(empty_stat)

        # THEN :
        assert out.numel() == 0
        assert out.shape == empty_stat.shape


class TestPatchMaskingAugmenter:

    @pytest.fixture
    def sample_batch_multiple(self):
        return torch.ones(4, 1, 64)

    def test__call__masks_in_contiguous_blocks_of_patch_len(self, sample_batch_multiple):
        # GIVEN :
        patch_len = 16
        augmenter = PatchMaskingAugmenter(mask_ratio=0.5, patch_len=patch_len)

        # WHEN :
        masked_batch = augmenter(sample_batch_multiple)

        # THEN :
        for i in range(sample_batch_multiple.size(0)):
            reshaped_signal = masked_batch[i, 0].view(-1, patch_len)

            for patch_idx in range(reshaped_signal.size(0)):
                patch_sum = reshaped_signal[patch_idx].sum().item()
                assert patch_sum == 0.0 or patch_sum == float(patch_len)

    def test__call__handles_non_multiple_sequence_length_gracefully(self):
        augmenter = PatchMaskingAugmenter(mask_ratio=1.0, patch_len=16)
        batch_odd_length = torch.ones(2, 2, 70)

        # WHEN :
        masked_batch = augmenter(batch_odd_length)

        # THEN :
        assert torch.all(masked_batch[:, :, :64] == 0.0)

        assert torch.all(masked_batch[:, :, 64:] == 1.0)