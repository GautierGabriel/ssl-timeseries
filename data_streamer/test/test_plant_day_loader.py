import logging

import numpy as np
import pandas as pd
import pytest

from vivml.SSL.data_streamer.plant_day import PlantDay
from vivml.SSL.data_streamer.plant_day_loader import DataLoader


class TestDataLoader:
    @pytest.fixture
    def plant_day(self):
        return PlantDay(
            channel_id="sensor-X",
            start_date="2026-05-20 00:00:00",
            stop_date="2026-05-20 23:59:59",
        )

    def test__clean_day_data__aligns_values_to_correct_seconds(self, plant_day):
        raw_data = pd.DataFrame(
            {
                "timestamp": [1779235202, 1779235205],
                "metric_value": [42.0, 84.0],
            }
        )
        loader = DataLoader(fetch_data=lambda _: raw_data)

        cleaned = loader.load_plant_day_data(plant_day)

        assert len(cleaned) == 86400
        assert cleaned["mV"].iloc[2] == 42.0
        assert cleaned["mV"].iloc[5] == 84.0
        assert np.isnan(cleaned["mV"].iloc[0])

    def test__clean_day_data__returns_nan_array_for_empty_input(self, plant_day):
        loader = DataLoader(fetch_data=lambda _: pd.DataFrame())

        cleaned = loader.load_plant_day_data(plant_day)

        assert len(cleaned) == 86400
        assert np.isnan(cleaned["mV"].values).all()

    def test__load_plant_day_data__propagates_source_errors(self, plant_day):
        def fail_to_fetch(_):
            raise RuntimeError("Data source unavailable")

        loader = DataLoader(fetch_data=fail_to_fetch, logger=logging.getLogger(__name__))

        with pytest.raises(RuntimeError, match="Data source unavailable"):
            loader.load_plant_day_data(plant_day)
