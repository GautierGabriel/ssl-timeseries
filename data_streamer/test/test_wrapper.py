import pytest
import torch
import numpy as np

from vivml.SSL.data_streamer.wrapper import stream_wrapper


class TestUniversalStreamWrapper:

    def test__universal_stream_wrapper__yields_x_none_and_metadata_for_unsupervised_task(self):
        # GIVEN
        device = torch.device("cpu")
        mock_metadata = {
            "channel_id": ["A", "B", "C", "D"],
            "label": [None, None, None, None]
        }
        mock_streamer = [
            (np.random.rand(4, 100), mock_metadata)
        ]

        # WHEN
        wrapper = stream_wrapper(mock_streamer, device)
        batch = next(wrapper)

        # THEN
        assert len(batch) == 3
        x, labels, metadata = batch

        assert isinstance(x, torch.Tensor)
        assert x.dtype == torch.float32
        assert x.shape == (4, 1, 100)

        assert labels is None
        assert metadata["channel_id"] == ["A", "B", "C", "D"]
        assert len(metadata["channel_id"]) == 4

    def test__universal_stream_wrapper__yields_x_labels_and_metadata_for_supervised_task(self):
        # GIVEN
        device = torch.device("cpu")
        mock_metadata = {
            "channel_id": ["A", "B", "C", "D"],
            "label": [1, 0, 2, 1]
        }
        mock_streamer = [
            (np.random.rand(4, 100), mock_metadata)
        ]

        # WHEN
        wrapper = stream_wrapper(mock_streamer, device)
        batch = next(wrapper)

        # THEN
        assert len(batch) == 3
        x, labels, metadata = batch

        assert isinstance(x, torch.Tensor)
        assert x.shape == (4, 1, 100)

        assert isinstance(labels, torch.Tensor)
        assert labels.dtype == torch.long
        assert labels.tolist() == [1, 0, 2, 1]

    def test__universal_stream_wrapper__converts_extra_channels_and_stats_to_tensors(self):
        # GIVEN
        device = torch.device("cpu")

        mock_metadata = {
            "extra_channels": {
                "HF": np.random.rand(2, 100)
            },
            "stds": {
                "raw": np.array([[0.5], [1.5]], dtype=np.float32)
            },
            "drift": np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        }

        mock_streamer = [
            (np.random.rand(2, 100), mock_metadata)
        ]

        # WHEN
        wrapper = stream_wrapper(mock_streamer, device)
        _, _, metadata = next(wrapper)

        # THEN
        assert isinstance(metadata["extra_channels"]["HF"], torch.Tensor)
        assert metadata["extra_channels"]["HF"].dtype == torch.float32

        assert isinstance(metadata["stds"]["raw"], torch.Tensor)
        assert metadata["stds"]["raw"].shape == (2, 1)

        assert isinstance(metadata["drift"], torch.Tensor)