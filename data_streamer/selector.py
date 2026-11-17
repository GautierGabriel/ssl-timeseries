from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple
import datetime

import pandas as pd

from vivml.SSL.data_streamer.plant_day import PlantDay


def get_optimized_intervals(
        start_date: datetime,
        end_date: datetime,
        window_size_sec: int,
        window_shift_sec: int ) -> List[Tuple[datetime, datetime]]:
    intervals = []
    current_start = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date)

    while current_start < final_end:
        theoretical_end = min(current_start + pd.Timedelta(days=1), final_end)
        duration_sec = (theoretical_end - current_start).total_seconds()
        n_windows = int(duration_sec // window_shift_sec)

        if n_windows > 0:
            last_start_sec = (n_windows - 1) * window_shift_sec
            optimized_end = (current_start +
                             pd.Timedelta(seconds=last_start_sec) +
                             pd.Timedelta(seconds=window_size_sec))

            intervals.append((current_start, optimized_end))

        current_start += pd.Timedelta(days=1)
    return intervals


class BaseSelector(ABC):
    @abstractmethod
    def get_plant_days(self) -> List[PlantDay]:
        pass


class PlantDaySelector(BaseSelector):
    def __init__(self, plant_days: Sequence[PlantDay]):
        self.plant_days = list(plant_days)

    def get_plant_days(self) -> List[PlantDay]:
        return list(self.plant_days)