import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict
from vivml.SSL.data_streamer.plant_day import PlantDay
from vivml.SSL.data_streamer.window import Window
from vivml.SSL.data_streamer.channel_extractor import BaseExtractor
from utils.utils import total_nans_perc, max_continuous_nan_perc



class WindowProcessor:
    def __init__(
            self,
            window_size_sec: int,
            window_shift_sec: int,
            extractors: Optional[List["BaseExtractor"]] = None,
            total_nans_perc_thresh: float = 0.3,
            max_continuous_nan_perc_thresh: float = 0.04,
            logger: Optional[logging.Logger] = None
    ):
        self.window_size_sec = window_size_sec
        self.window_shift_sec = window_shift_sec
        self.extractors = extractors or []
        self.total_nans_perc_thresh = total_nans_perc_thresh
        self.max_continuous_nan_perc_thresh = max_continuous_nan_perc_thresh
        self.logger = logger or logging.getLogger(__name__)

    def split_data_into_windows(
            self, data: pd.DataFrame, plant_day: PlantDay
    ) -> List[Window]:

        raw_values = data["mV"].values
        timestamps = data.index.values
        total_len = len(raw_values)

        if total_len == 0 or np.isnan(raw_values).all():
            self.logger.debug(f"Day rejected: {plant_day.channel_id} (no data or 100% NaNs)")
            return []

        day_clean = pd.Series(raw_values).interpolate(limit_direction='both').values
        n_points = len(day_clean)

        first_val = np.mean(day_clean[:10])
        last_val = np.mean(day_clean[-10:])

        idx_first = (10 - 1) / 2.0
        idx_last = (n_points - 1) - (10 - 1) / 2.0

        if idx_last > idx_first:
            drift_a = (last_val - first_val) / (idx_last - idx_first)
            drift_b = first_val - drift_a * idx_first
        else:
            drift_a = 0.0
            drift_b = first_val

        drift_params = [drift_a, drift_b]

        day_channels_data = {}
        stds_dict = {
            "raw": np.std(day_clean) if np.std(day_clean) > 0 else 1e-6,
        }

        for extractor in self.extractors:
            extracted_channels = extractor.apply(signal=day_clean, timestamps=timestamps)

            for ch_name, ch_data in extracted_channels.items():
                day_channels_data[ch_name] = ch_data

                if ch_name in ["heure_sin", "heure_cos", "date_sin", "date_cos"]:
                    stds_dict[ch_name] = 1.0
                else:
                    std_val = np.std(ch_data)
                    stds_dict[ch_name] = std_val if std_val > 0 else 1e-6

        windows = []
        size = self.window_size_sec
        shift = self.window_shift_sec

        for start in range(0, total_len - size + 1, shift):
            end = start + size
            raw_window = raw_values[start:end]

            if len(raw_window) < size:
                continue

            if np.isnan(raw_window).any():
                if (total_nans_perc(raw_window) < self.total_nans_perc_thresh and
                        max_continuous_nan_perc(raw_window) < self.max_continuous_nan_perc_thresh):
                    raw_window = day_clean[start:end]
                else:
                    self.logger.debug(
                        f"[{plant_day.channel_id}] Window {start}-{end} rejected: "
                        f"Total NaNs={total_nans_perc(raw_window):.2%}, Max Continuous={max_continuous_nan_perc(raw_window):.2%}"
                    )
                    continue

            window_channels = {
                name: channel_data[start:end]
                for name, channel_data in day_channels_data.items()
            }

            stats = {
                "drift": drift_params,
                "stds": stds_dict,
                "start_idx": start
            }

            windows.append(
                Window(
                    data=raw_window,
                    extra_channels=window_channels,
                    stats=stats,
                    channel_id=plant_day.channel_id,
                    timestamp=timestamps[end - 1],
                    trial_metadata=plant_day.trial_metadata,
                    label=plant_day.label
                )
            )

        return windows