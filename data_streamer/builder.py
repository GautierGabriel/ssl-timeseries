from typing import Optional, Tuple

from vivml.SSL.data_streamer.plant_day_loader import DataLoader
from vivml.SSL.data_streamer.window_processor import WindowProcessor
from vivml.SSL.data_streamer.batch_streamer import BatchStreamer
from vivml.SSL.data_streamer.channel_extractor import FrequencyExtractor, TimeExtractor
from vivml.SSL.data_streamer.selector import PlantDaySelector
from vivml.SSL.data_streamer.source import DataSource


def get_dataloaders(
        config: dict,
        logger,
        data_source: Optional[DataSource] = None,
        val_max_plant_days_override: Optional[int] = None,
        eval_split: str = "val",
) -> Tuple['BatchStreamer', 'BatchStreamer']:
    if data_source is None:
        raise ValueError(
            "Provide a DataSource implementation from your authorized data environment."
        )

    data_cfg = config["data"].copy()
    input_features_cfg = data_cfg.pop("input_features", {})
    train_plant_days = list(data_source.get_plant_days("train", data_cfg))
    val_plant_days = list(data_source.get_plant_days(eval_split, data_cfg))
    loader = DataLoader(fetch_data=data_source.load_raw_data, logger=logger)

    sampling_rate_sec = data_cfg.pop("sampling_rate_sec", None)
    extractors = []
    if sampling_rate_sec and input_features_cfg:
        sampling_rate_hz = 1.0 / sampling_rate_sec

        for ch_name, f_cfg in input_features_cfg.items():
            if f_cfg is None:
                continue

            ext_type = f_cfg.get("extractor_type")

            if ext_type == "frequency":
                cutoff_freq_hz = 1.0 / f_cfg["cutoff_period_sec"]

                extractor = FrequencyExtractor(
                    channel_name=ch_name,
                    sampling_rate_hz=sampling_rate_hz,
                    cutoff_freq_hz=cutoff_freq_hz,
                    filter_type=f_cfg["filter_type"],
                    order=f_cfg.get("order", 4)
                )
                extractors.append(extractor)

            elif ext_type == "time":
                extractor = TimeExtractor(
                    time_type=f_cfg["time_type"]
                )
                extractors.append(extractor)

    processor = WindowProcessor(
        window_size_sec=data_cfg["window_len"],
        window_shift_sec=data_cfg["window_shift"],
        extractors=extractors,
        total_nans_perc_thresh=0.3,
        max_continuous_nan_perc_thresh=0.04,
        logger=logger
    )

    batch_size = data_cfg.pop("batch_size", 64)
    max_queue_size = data_cfg.pop("max_queue_size", 1000)
    num_fetch_workers = data_cfg.pop("num_fetch_workers", 8)
    if val_max_plant_days_override is not None:
        val_plant_days = val_plant_days[:val_max_plant_days_override]

    train_streamer = BatchStreamer(
        selector=PlantDaySelector(train_plant_days),
        loader=loader,
        processor=processor,
        batch_size=batch_size,
        max_queue_size=max_queue_size,
        num_fetch_workers=num_fetch_workers,
        shuffle_batch=data_cfg.get("shuffle", True),
        logger=logger,
        domain_mapping=data_cfg.get("domain_mapping"),
    )

    val_streamer = BatchStreamer(
        selector=PlantDaySelector(val_plant_days),
        loader=loader,
        processor=processor,
        batch_size=batch_size,
        max_queue_size=max_queue_size,
        num_fetch_workers=num_fetch_workers,
        shuffle_batch=False,
        logger=logger,
        domain_mapping=data_cfg.get("domain_mapping"),
    )

    return train_streamer, val_streamer