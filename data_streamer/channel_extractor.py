from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


class BaseExtractor(ABC):
    @abstractmethod
    def apply(self, signal: np.ndarray, timestamps: np.ndarray) -> dict:
        pass

class FrequencyExtractor(BaseExtractor):
    def __init__(
            self,
            sampling_rate_hz: float,
            cutoff_freq_hz: float,
            channel_name: str,
            filter_type: str = "highpass",
            order: int = 4
    ):
        self.sampling_rate_hz = sampling_rate_hz
        self.cutoff_freq_hz = cutoff_freq_hz
        self.channel_name = channel_name
        self.filter_type = filter_type.lower()
        self.order = order

        if self.filter_type not in ["highpass", "lowpass"]:
            raise ValueError("filter_type should be 'highpass' or 'lowpass'")

        nyquist = 0.5 * self.sampling_rate_hz
        normal_cutoff = self.cutoff_freq_hz / nyquist
        btype = 'high' if self.filter_type == "highpass" else 'low'
        self.b, self.a = butter(self.order, normal_cutoff, btype=btype, analog=False)

    def apply(self, signal: np.ndarray, timestamps: np.ndarray) -> dict:
        filtered = filtfilt(self.b, self.a, signal)
        return {self.channel_name: filtered}

class TimeExtractor(BaseExtractor):
    def __init__(self, time_type: str):
        self.time_type = time_type.lower()
        if self.time_type not in ["date", "time"]:
            raise ValueError("time_type should be 'date' or 'time'")

    def apply(self, signal: np.ndarray, timestamps: np.ndarray) -> dict:

        dt_index = pd.to_datetime(timestamps)

        if self.time_type == "date":
            jour_annee = dt_index.dayofyear
            return {
                "date_sin": np.sin(2 * np.pi * jour_annee / 365.25),
                "date_cos": np.cos(2 * np.pi * jour_annee / 365.25)
            }

        elif self.time_type == "time":
            heures_decimales = dt_index.hour + (dt_index.minute / 60.0) + (dt_index.second / 3600.0)
            return {
                "time_sin": np.sin(2 * np.pi * heures_decimales / 24.0),
                "time_cos": np.cos(2 * np.pi * heures_decimales / 24.0)
            }