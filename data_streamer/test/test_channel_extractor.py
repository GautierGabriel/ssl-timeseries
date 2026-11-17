import pytest
import numpy as np
from vivml.SSL.data_streamer.channel_extractor import FrequencyExtractor

class TestFrequencyExtractor:

    @pytest.fixture
    def sampling_rate_hz(self):
        return 1.0 / 60.0

    @pytest.fixture
    def cutoff_freq_hz(self):
        return 1.0 / 7200.0

    def test__apply__hf_removes_slow_macro_cycle(self, sampling_rate_hz, cutoff_freq_hz):
        t = np.arange(0, 86400, 60)
        macro_cycle = 100 * np.sin(2 * np.pi * t / 86400)
        micro_stress = 2 * np.sin(2 * np.pi * t / 600)
        signal_brut = macro_cycle + micro_stress

        extractor = FrequencyExtractor(
            channel_name="HF",
            sampling_rate_hz=sampling_rate_hz,
            cutoff_freq_hz=cutoff_freq_hz,
            filter_type="highpass",
            order=4
        )

        extracted_channels = extractor.apply(signal=signal_brut, timestamps=t)
        filtered_signal = extracted_channels["HF"]

        assert np.max(np.abs(filtered_signal)) < 5.0
        assert np.std(filtered_signal) > 0.5

    def test__apply__invalid_filter_type_raises_value_error(self, sampling_rate_hz, cutoff_freq_hz):
        with pytest.raises(ValueError, match="filter_type should be 'highpass' or 'lowpass'"):
            FrequencyExtractor(
                channel_name="Invalid_Channel",
                sampling_rate_hz=sampling_rate_hz,
                cutoff_freq_hz=cutoff_freq_hz,
                filter_type="INVALID"
            )