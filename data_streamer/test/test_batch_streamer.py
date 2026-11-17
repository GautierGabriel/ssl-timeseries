import pytest
import numpy as np
import pandas as pd
import logging
from unittest.mock import MagicMock
from vivml.SSL.data_streamer.plant_day import PlantDay
from vivml.SSL.data_streamer.window import Window
from vivml.SSL.data_streamer.batch_streamer import BatchStreamer


class TestBatchStreamer:

    @pytest.fixture
    def mock_components(self):
        selector = MagicMock()
        loader = MagicMock()
        processor = MagicMock()

        plant_day = PlantDay("ch-test", "2026-05-20", "2026-05-21", {}, None)
        selector.get_plant_days.return_value = [plant_day]
        loader.load_plant_day_data.return_value = pd.DataFrame()

        return selector, loader, processor, plant_day

    def test__streamer__creates_correct_sized_batches(self, mock_components):
        selector, loader, processor, plant_day = mock_components

        win1 = Window(
            data=np.ones(10),
            channel_id="ch-test",
            timestamp=1000,
            trial_metadata={},
            label=None,
            extra_channels={},
            stats={}
        )
        win2 = Window(
            data=np.ones(10) * 2,
            channel_id="ch-test",
            timestamp=2000,
            trial_metadata={},
            label=None,
            extra_channels={},
            stats={}
        )
        win3 = Window(
            data=np.ones(10) * 3,
            channel_id="ch-test",
            timestamp=3000,
            trial_metadata={},
            label=None,
            extra_channels={},
            stats={}
        )
        processor.split_data_into_windows.return_value = [win1, win2, win3]

        streamer = BatchStreamer(
            selector=selector,
            loader=loader,
            processor=processor,
            batch_size=2,
            shuffle_batch=False,
            logger=logging.getLogger(__name__)
        )

        batches = list(streamer)

        assert len(batches) == 1

        first_batch_data, first_batch_meta = batches[0]
        assert first_batch_data.shape == (2, 10)
        assert first_batch_meta["timestamp"][0] == 1000
        assert first_batch_meta["timestamp"][1] == 2000

    def test__streamer__terminates_when_stopped_early(self, mock_components):
        selector, loader, processor, plant_day = mock_components
        infinite_windows = [
            Window(
                data=np.zeros(10),
                channel_id="ch-test",
                timestamp=i,
                trial_metadata={},
                label=None,
                extra_channels={},
                stats={}
            )
            for i in range(100)
        ]
        processor.split_data_into_windows.return_value = infinite_windows

        streamer = BatchStreamer(
            selector=selector,
            loader=loader,
            processor=processor,
            batch_size=4,
            logger=logging.getLogger(__name__)
        )

        for batch_data, _ in streamer:
            assert batch_data.shape == (4, 10)
            streamer.stop()
            break

        assert not streamer.producer_thread.is_alive()