import dataclasses
import datetime
from typing import Optional

@dataclasses.dataclass
class PlantDay:
    channel_id: str
    start_date: datetime.date
    stop_date: datetime.date
    trial_metadata: dict = dataclasses.field(default_factory=dict)
    label: Optional[int] = None