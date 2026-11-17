import datetime

from vivml.SSL.data_streamer.plant_day import PlantDay
from vivml.SSL.data_streamer.selector import PlantDaySelector, get_optimized_intervals


class TestPlantDaySelector:
    def test__get_optimized_intervals__bounds_intervals_correctly(self):
        start = datetime.datetime(2026, 5, 20, 0, 0, 0)
        end = datetime.datetime(2026, 5, 22, 0, 0, 0)

        intervals = get_optimized_intervals(start, end, 600, 300)

        assert len(intervals) == 2
        assert intervals[0][0] == start
        assert intervals[1][0] == datetime.datetime(2026, 5, 21, 0, 0, 0)

    def test__get_plant_days__returns_injected_segments(self):
        expected = [PlantDay("channel-1", "2026-05-20", "2026-05-21")]

        selector = PlantDaySelector(expected)

        assert selector.get_plant_days() == expected
