import dataclasses
import multiprocessing.managers
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclasses.dataclass
class Window:
    data: np.ndarray
    channel_id: str
    timestamp: datetime
    trial_metadata: dict = dataclasses.field(default_factory=dict)
    label: Optional[int] = None
    extra_channels: Optional[dict] = None
    stats: Optional[dict] = None