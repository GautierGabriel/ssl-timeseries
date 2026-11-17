import logging
from typing import Callable, Optional

import pandas as pd
from vivml.SSL.data_streamer.plant_day import PlantDay
import numpy as np


class DataLoader:
    def __init__(
        self,
        fetch_data: Callable[[PlantDay], Optional[pd.DataFrame]],
        logger: Optional[logging.Logger] = None
    ):
        self.fetch_data = fetch_data
        self.logger = logger or logging.getLogger(__name__)

    def load_plant_day_data(self, plant_day: PlantDay) -> pd.DataFrame:
        try:
            day_data = self.fetch_data(plant_day)
            if day_data is None:
                self.logger.warning(
                    "No data returned for %s at %s",
                    plant_day.channel_id,
                    plant_day.start_date,
                )
                return pd.DataFrame()
            return self.clean_day_data(day_data, plant_day)
        except Exception:
            self.logger.exception(
                "Failed loading data for %s at %s",
                plant_day.channel_id,
                plant_day.start_date,
            )
            raise

    def clean_day_data(self, day_data: pd.DataFrame, plant_day: 'PlantDay') -> pd.DataFrame:
        if day_data.empty or "timestamp" not in day_data.columns or "metric_value" not in day_data.columns:
            return pd.DataFrame({"mV": np.full(86400, np.nan)})

        first_ts = day_data["timestamp"].iloc[0]
        ts_unit = "ms" if first_ts > 5e10 else "s"

        timestamps = pd.to_datetime(day_data["timestamp"], unit=ts_unit)
        values = day_data["metric_value"].values

        self.logger.debug(f"[{plant_day.channel_id}] Missing records detected: {np.isnan(values).sum()}/86400")

        seconds_in_day = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
        seconds_indices = seconds_in_day.values

        aligned_values = np.full(86400, np.nan)
        valid_mask = (seconds_indices >= 0) & (seconds_indices < 86400)
        aligned_values[seconds_indices[valid_mask]] = values[valid_mask]

        start_dt = pd.to_datetime(plant_day.start_date).normalize()
        target_index = pd.date_range(start=start_dt, periods=86400, freq="1s")

        cleaned_df = pd.DataFrame({"mV": aligned_values}, index=target_index)
        return cleaned_df