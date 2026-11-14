from contextlib import contextmanager
import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import psutil


@contextmanager
def measure(label: str, interval: float = 0.025):
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    peak_children = 0
    stop = Event()
    result = {"stage": label}

    def sample():
        nonlocal peak, peak_children
        rss = process.memory_info().rss
        child_rss = 0
        for child in process.children(recursive=True):
            try:
                info = child.memory_info()
                child_rss += getattr(info, "peak_wset", info.rss)
                rss += info.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak = max(peak, rss)
        peak_children = max(peak_children, child_rss)

    def monitor():
        while not stop.wait(interval):
            sample()

    worker = Thread(target=monitor, daemon=True)
    started = perf_counter()
    worker.start()
    try:
        yield result
    finally:
        elapsed = perf_counter() - started
        stop.set()
        worker.join()
        sample()
        result.update(seconds=elapsed, peak_rss_bytes=peak, baseline_rss_bytes=baseline,
                      peak_child_rss_bytes=peak_children, rss_sample_interval_seconds=interval)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
