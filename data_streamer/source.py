from typing import Mapping, Optional, Protocol, Sequence

import pandas as pd

from vivml.SSL.data_streamer.plant_day import PlantDay


class DataSource(Protocol):
    def get_plant_days(
        self, split: str, config: Mapping[str, object]
    ) -> Sequence[PlantDay]:
        """Return segments for the "train", "val", or "test" split."""
        ...

    def load_raw_data(self, plant_day: PlantDay) -> Optional[pd.DataFrame]:
        """Return timestamp and metric_value columns for one segment."""
        ...
