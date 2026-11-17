import logging
from unittest.mock import MagicMock

from vivml.SSL.data_streamer.batch_streamer import BatchStreamer
from vivml.SSL.data_streamer.builder import get_dataloaders
from vivml.SSL.data_streamer.channel_extractor import FrequencyExtractor, TimeExtractor
from vivml.SSL.data_streamer.plant_day import PlantDay


class TestDataBuilder:
    def test__get_dataloaders__builds_streamers_from_injected_source(self):
        source = MagicMock()
        source.get_plant_days.side_effect = [
            [PlantDay("train-channel", "2026-05-20", "2026-05-21")],
            [PlantDay("val-channel", "2026-05-21", "2026-05-22")],
        ]
        config = {
            "data": {
                "window_len": 600,
                "window_shift": 300,
                "shuffle": True,
                "batch_size": 32,
                "max_queue_size": 100,
                "input_features": {"raw": None},
            }
        }

        train_streamer, val_streamer = get_dataloaders(
            config, logging.getLogger(__name__), source
        )

        assert source.get_plant_days.call_args_list[0].args[0] == "train"
        assert source.get_plant_days.call_args_list[1].args[0] == "val"
        assert isinstance(train_streamer, BatchStreamer)
        assert isinstance(val_streamer, BatchStreamer)
        assert train_streamer.batch_size == 32
        assert val_streamer.shuffle_batch is False
        assert train_streamer.selector.get_plant_days()[0].channel_id == "train-channel"
        assert val_streamer.selector.get_plant_days()[0].channel_id == "val-channel"
        assert train_streamer.processor.extractors == []

    def test__get_dataloaders__parses_input_feature_config(self):
        source = MagicMock()
        source.get_plant_days.return_value = []
        config = {
            "data": {
                "window_len": 600,
                "window_shift": 300,
                "batch_size": 32,
                "max_queue_size": 100,
                "sampling_rate_sec": 60,
                "input_features": {
                    "HF": {
                        "extractor_type": "frequency",
                        "filter_type": "highpass",
                        "cutoff_period_sec": 7200,
                        "order": 4,
                    },
                    "clock": {"extractor_type": "time", "time_type": "time"},
                },
            }
        }

        train_streamer, _ = get_dataloaders(
            config, logging.getLogger(__name__), source
        )

        extractors = train_streamer.processor.extractors
        assert len(extractors) == 2
        assert isinstance(extractors[0], FrequencyExtractor)
        assert extractors[0].channel_name == "HF"
        assert extractors[0].filter_type == "highpass"
        assert extractors[0].order == 4
        assert extractors[0].sampling_rate_hz == 1.0 / 60.0
        assert extractors[0].cutoff_freq_hz == 1.0 / 7200.0
        assert isinstance(extractors[1], TimeExtractor)
        assert extractors[1].time_type == "time"

    def test__get_dataloaders__allows_a_held_out_test_split(self):
        source = MagicMock()
        source.get_plant_days.return_value = []
        config = {
            "data": {
                "window_len": 600,
                "window_shift": 300,
                "input_features": {},
            }
        }

        get_dataloaders(
            config,
            logging.getLogger(__name__),
            source,
            eval_split="test",
        )

        assert [call.args[0] for call in source.get_plant_days.call_args_list] == [
            "train",
            "test",
        ]
