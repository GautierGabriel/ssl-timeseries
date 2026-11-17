import pytest
import numpy as np
import pandas as pd
import logging

from vivml.SSL.data_streamer.plant_day import PlantDay
from vivml.SSL.data_streamer.window_processor import WindowProcessor
from vivml.SSL.data_streamer.channel_extractor import FrequencyExtractor


class TestWindowProcessor:

    @pytest.fixture
    def hf_extractor(self):
        return FrequencyExtractor(
            channel_name="HF",
            sampling_rate_hz=1.0,
            cutoff_freq_hz=1.0 / 120.0,
            filter_type="highpass"
        )

    @pytest.fixture
    def processor(self):
        return WindowProcessor(
            window_size_sec=600,
            window_shift_sec=300,
            total_nans_perc_thresh=0.3,
            max_continuous_nan_perc_thresh=0.04,
            logger=logging.getLogger(__name__)
        )

    @pytest.fixture
    def processor_multi_channel(self, hf_extractor):
        return WindowProcessor(
            window_size_sec=600,
            window_shift_sec=300,
            extractors=[hf_extractor],
            total_nans_perc_thresh=0.3,
            max_continuous_nan_perc_thresh=0.04,
            logger=logging.getLogger(__name__)
        )

    @pytest.fixture
    def dummy_plant_day(self):
        return PlantDay(
            channel_id="test-channel-1",
            start_date="2026-05-20 00:00:00",
            stop_date="2026-05-20 23:59:59",
            trial_metadata={"experiment": "test"},
            label=None
        )

    def test__split_data_into_windows__rejects_total_nan_day(self, processor, dummy_plant_day):
        target_index = pd.date_range(start="2026-05-20", periods=86400, freq="1s")
        df_nan = pd.DataFrame({"mV": np.full(86400, np.nan)}, index=target_index)

        windows = processor.split_data_into_windows(df_nan, dummy_plant_day)
        assert len(windows) == 0

    def test__split_data_into_windows__returns_correct_end_timestamps(self, processor, dummy_plant_day):
        target_index = pd.date_range(start="2026-05-20 00:00:00", periods=3600, freq="1s")
        df_clean = pd.DataFrame({"mV": np.random.rand(3600)}, index=target_index)

        windows = processor.split_data_into_windows(df_clean, dummy_plant_day)

        assert len(windows) == 11
        assert windows[0].timestamp == target_index.values[599]

    def test__split_data_into_windows__interpolates_acceptable_nans(self, processor, dummy_plant_day):
        target_index = pd.date_range(start="2026-05-20 00:00:00", periods=600, freq="1s")
        signal = np.random.rand(600)
        signal[100:110] = np.nan
        df_dirty = pd.DataFrame({"mV": signal}, index=target_index)

        windows = processor.split_data_into_windows(df_dirty, dummy_plant_day)

        assert len(windows) == 1
        assert not np.isnan(windows[0].data).any()

    def test__split_data_into_windows__calculates_robust_drift_with_outlier_protection(self, processor,
                                                                                       dummy_plant_day):
        target_index = pd.date_range(start="2026-05-20 00:00:00", periods=1200, freq="1s")
        signal = np.full(1200, 10.0)
        signal[0] = 999.0
        df_drift = pd.DataFrame({"mV": signal}, index=target_index)

        windows = processor.split_data_into_windows(df_drift, dummy_plant_day)
        drift_slope = windows[0].stats["drift"][0]

        assert abs(drift_slope) < 1.0

    def test__split_data_into_windows__populates_extra_channels_and_stds(self, processor_multi_channel,
                                                                         dummy_plant_day):
        target_index = pd.date_range(start="2026-05-20 00:00:00", periods=3600, freq="1s")
        df_clean = pd.DataFrame({"mV": np.random.rand(3600)}, index=target_index)

        windows = processor_multi_channel.split_data_into_windows(df_clean, dummy_plant_day)

        assert len(windows) > 0
        first_window = windows[0]

        assert "HF" in first_window.extra_channels
        assert first_window.extra_channels["HF"].shape == (600,)

        assert "stds" in first_window.stats
        assert "raw" in first_window.stats["stds"]
        assert "HF" in first_window.stats["stds"]

        assert first_window.stats["stds"]["raw"] != first_window.stats["stds"]["HF"]