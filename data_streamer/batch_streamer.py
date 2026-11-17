import numpy as np
import threading
import queue
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from typing import List, Dict, Iterator, Tuple, Optional
from vivml.SSL.data_streamer.window_processor import WindowProcessor
from vivml.SSL.data_streamer.window import Window
from tqdm import tqdm
import torch

class BatchStreamer:
    def __init__(
            self,
            selector: "BaseSelector",
            loader: "DataLoader",
            processor: "WindowProcessor",
            batch_size: int,
            shuffle_batch: bool = True,
            max_queue_size: int = 1000,
            num_fetch_workers: int = 8,
            shuffle_buffer_batches: int = 3,
            logger: Optional[logging.Logger] = None,
            domain_mapping: Optional[Dict[str, int]] = None,
    ):
        self.selector = selector
        self.loader = loader
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch
        self.num_fetch_workers = num_fetch_workers
        self.shuffle_buffer_batches = shuffle_buffer_batches

        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.window_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_signal = threading.Event()
        self.producer_thread = None

        self.plant_days_cache = None

        self.channel_map = domain_mapping or {}
        self.unknown_idx = self.channel_map.get("UNKNOWN_CHANNEL", 0)

    def _create_batch(self, window_list: List) -> Tuple[np.ndarray, Dict]:
        batch_windows = np.stack([w.data for w in window_list])

        domain_ids = [
            self.channel_map.get(str(w.channel_id), self.unknown_idx)
            for w in window_list
        ]

        batch_metadata = {
            "channel_id": [w.channel_id for w in window_list],
            "timestamp": [w.timestamp for w in window_list],
            "label": [w.label for w in window_list],
            "trial_metadata": [w.trial_metadata for w in window_list],
            "extra_channels": {},
            "stds": {},
            "domain_labels": torch.tensor(domain_ids, dtype=torch.long)
        }

        first_window = window_list[0]

        if hasattr(first_window, 'extra_channels') and first_window.extra_channels:
            for ch in first_window.extra_channels.keys():
                batch_metadata["extra_channels"][ch] = np.stack(
                    [w.extra_channels[ch] for w in window_list]
                )

        if hasattr(first_window, 'stats') and first_window.stats:
            if "stds" in first_window.stats:
                for ch in first_window.stats["stds"].keys():
                    batch_metadata["stds"][ch] = np.array(
                        [w.stats["stds"][ch] for w in window_list],
                        dtype=np.float32
                    ).reshape(-1, 1)

            if "drift" in first_window.stats:
                batch_metadata["drift"] = np.stack(
                    [w.stats["drift"] for w in window_list]
                )

            if "start_idx" in first_window.stats:
                batch_metadata["start_idx"] = np.array(
                    [w.stats["start_idx"] for w in window_list],
                    dtype=np.int32
                )

        return batch_windows, batch_metadata

    def __iter__(self) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        self.stop_signal.clear()

        while not self.window_queue.empty():
            try:
                self.window_queue.get_nowait()
            except queue.Empty:
                break

        self.producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.producer_thread.start()

        # Plant-days are shuffled (see _producer_loop), but windows *within* a day are
        # pushed to the queue in chronological, unshuffled order - one contiguous block
        # per day. Pulled straight off the FIFO queue, a batch ends up dominated by just
        # a handful of days (heavily-overlapping, near-duplicate windows), which starves
        # contrastive learning of real negatives. This shuffle buffer accumulates several
        # batches' worth of windows, shuffles them together, then draws one batch at a
        # time from that mixed pool, so each batch actually contains a real cross-section
        # of different plant-days instead of a few contiguous blocks.
        # Only pay for the larger buffer/mixing when shuffling is actually wanted -
        # val streams use shuffle_batch=False for deterministic, reproducible ordering,
        # and should behave exactly as before (yield as soon as one batch is available).
        shuffle_buffer = []
        shuffle_buffer_size = self.batch_size * self.shuffle_buffer_batches if self.shuffle_batch else self.batch_size
        normal_exit = False
        total_wait_time = 0.0
        stall_warn_threshold_sec = 0.1
        pass_start = time.time()
        stream_ended = False

        try:
            while True:
                while len(shuffle_buffer) < shuffle_buffer_size and not stream_ended:
                    wait_start = time.time()
                    try:
                        item = self.window_queue.get(timeout=5)
                    except queue.Empty:
                        total_wait_time += time.time() - wait_start
                        if not self.producer_thread.is_alive():
                            stream_ended = True
                            break
                        self.logger.warning(
                            "Window queue empty for 5s while producer is still running - "
                            "training is stalled waiting on data fetch from MDM."
                        )
                        continue

                    wait_time = time.time() - wait_start
                    total_wait_time += wait_time
                    if wait_time > stall_warn_threshold_sec:
                        self.logger.warning(
                            f"Waited {wait_time:.2f}s for the next window "
                            f"(qsize={self.window_queue.qsize()}/{self.window_queue.maxsize}) - "
                            f"consumer is faster than the MDM data fetch here."
                        )

                    if item is None:
                        stream_ended = True
                        break

                    shuffle_buffer.append(item)

                if len(shuffle_buffer) < self.batch_size:
                    normal_exit = True
                    break

                if self.shuffle_batch:
                    random.shuffle(shuffle_buffer)
                batch_items = shuffle_buffer[:self.batch_size]
                shuffle_buffer = shuffle_buffer[self.batch_size:]
                yield self._create_batch(batch_items)
        finally:
            elapsed = time.time() - pass_start
            if elapsed > 0:
                self.logger.info(
                    f"Streaming pass finished: waited on data fetch for {total_wait_time:.1f}s "
                    f"of {elapsed:.1f}s total ({100 * total_wait_time / elapsed:.1f}% of the pass)."
                )
            self.stop()

    def _producer_loop(self) -> None:
        try:
            if self.plant_days_cache is None:
                self.plant_days_cache = self.selector.get_plant_days()

            current_plant_days = list(self.plant_days_cache)

            if self.shuffle_batch:
                random.shuffle(current_plant_days)

            pbar = tqdm(total=len(current_plant_days), desc="Streaming days", unit="day", leave=False)

            # Only the network fetch (I/O-bound, releases the GIL while blocked) runs
            # concurrently across threads. Windowing is CPU-bound Python and stays
            # single-threaded here, exactly as before, to avoid GIL contention with
            # the consumer thread.
            with ThreadPoolExecutor(max_workers=self.num_fetch_workers) as executor:
                day_iter = iter(current_plant_days)
                in_flight = {}

                for _ in range(self.num_fetch_workers):
                    plant_day = next(day_iter, None)
                    if plant_day is None:
                        break
                    in_flight[executor.submit(self.loader.load_plant_day_data, plant_day)] = plant_day

                while in_flight and not self.stop_signal.is_set():
                    done, _ = wait(in_flight.keys(), timeout=0.5, return_when=FIRST_COMPLETED)

                    for future in done:
                        plant_day = in_flight.pop(future)
                        pbar.update(1)

                        try:
                            day_data = future.result()
                            windows = self.processor.split_data_into_windows(day_data, plant_day)

                            for window in windows:
                                if self.stop_signal.is_set():
                                    break
                                while not self.stop_signal.is_set():
                                    try:
                                        self.window_queue.put(window, timeout=0.1)
                                        break
                                    except queue.Full:
                                        continue
                        except Exception as e:
                            self.logger.warning(f"Skipping processing {plant_day.channel_id}: error: {e}")

                        if not self.stop_signal.is_set():
                            next_day = next(day_iter, None)
                            if next_day is not None:
                                in_flight[executor.submit(self.loader.load_plant_day_data, next_day)] = next_day

            pbar.close()

        except Exception as fatal_error:
            self.logger.error(f"Fatal crash inside producer loop: {fatal_error}")
        finally:
            self.window_queue.put(None)

    def stop(self):
        self.stop_signal.set()
        while not self.window_queue.empty():
            try:
                self.window_queue.get_nowait()
            except queue.Empty:
                break

        if self.producer_thread and self.producer_thread.is_alive() and threading.current_thread() != self.producer_thread:
            self.producer_thread.join(timeout=2.0)