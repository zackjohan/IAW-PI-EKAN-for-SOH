"""Shared logging, device selection, timing, and memory accounting."""

import gc
import logging
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch


def select_device():
    requested = os.environ.get("PI_EKAN_DEVICE", "auto")
    if requested == "cpu" or not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.empty_cache()
        return "cuda"
    except RuntimeError as exc:
        logging.getLogger(__name__).warning("CUDA unavailable; using CPU: %s", exc)
        return "cpu"


device = select_device()


def get_logger(log_path=None, name="research"):
    """Create an isolated run logger, closing any previous handlers."""
    logger = logging.getLogger(
        f'{name}:{Path(log_path).absolute() if log_path else "console"}'
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logger(logger)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


class ComputationalTracker:
    """Synchronized epoch timing, sampled CPU RSS, and CUDA allocator peaks.

    CPU memory is sampled at epoch boundaries; it is not a continuous peak.
    Training elapsed time includes validation/test work inside the training loop.
    """

    def __init__(self, model_name=None):
        self.model_name = model_name
        self.epoch_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.experiment_start_time = None
        self.training_start_time = None
        self.epoch_start_time = None
        self.training_end_time = None

    def _sync(self):
        if device == "cuda":
            torch.cuda.synchronize()

    def start_experiment(self):
        self._sync()
        self.experiment_start_time = time.perf_counter()

    def start_training(self):
        self._sync()
        self.training_start_time = time.perf_counter()
        self.training_end_time = None

    def stop_training(self):
        self._sync()
        self.training_end_time = time.perf_counter()

    def start_epoch(self):
        self._sync()
        self.epoch_start_time = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def end_epoch(self):
        self._sync()
        if self.epoch_start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        self.cpu_memory_usage.append(psutil.Process().memory_info().rss / 1024**3)
        if device == "cuda":
            self.gpu_memory_usage.append(
                {
                    "allocated": torch.cuda.memory_allocated() / 1024**3,
                    "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
                    "reserved": torch.cuda.memory_reserved() / 1024**3,
                }
            )
        return elapsed

    def get_training_time(self):
        if self.training_start_time is None:
            return 0.0
        return (
            self.training_end_time or time.perf_counter()
        ) - self.training_start_time

    def get_experiment_time(self):
        return (
            time.perf_counter() - self.experiment_start_time
            if self.experiment_start_time
            else 0.0
        )

    def get_average_epoch_time(self):
        return float(np.mean(self.epoch_times)) if self.epoch_times else 0.0

    def get_total_training_time_estimate(self, total_epochs):
        return self.get_average_epoch_time() * total_epochs

    def get_memory_stats(self):
        stats = {
            "cpu_memory_avg": (
                float(np.mean(self.cpu_memory_usage)) if self.cpu_memory_usage else 0.0
            ),
            "cpu_memory_max": max(self.cpu_memory_usage, default=0.0),
        }
        if self.gpu_memory_usage:
            stats.update(
                {
                    "gpu_memory_avg_allocated": float(
                        np.mean([m["allocated"] for m in self.gpu_memory_usage])
                    ),
                    "gpu_memory_max_allocated": max(
                        m["max_allocated"] for m in self.gpu_memory_usage
                    ),
                    "gpu_memory_avg_reserved": float(
                        np.mean([m["reserved"] for m in self.gpu_memory_usage])
                    ),
                    "gpu_memory_max_reserved": max(
                        m["reserved"] for m in self.gpu_memory_usage
                    ),
                }
            )
        return stats

    def reset_memory_tracking(self):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()


def benchmark_forward(model, inputs, warmup=5, repeats=50):
    """Time the SOH prediction network without data-transfer overhead."""
    model.eval()
    is_cuda = inputs.device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
        if is_cuda:
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return {
        "inference_batch_size_measured": len(inputs),
        "inference_num_timing_runs": repeats,
        "inference_avg_pass_ms": elapsed * 1000,
        "inference_time_ms_per_1000": elapsed * 1_000_000 / len(inputs),
    }
