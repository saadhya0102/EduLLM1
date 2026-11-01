"""Small dependency-free progress and ETA helpers for long local jobs."""
from __future__ import annotations

import time


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class Progress:
    def __init__(
            self, label: str, total: int, unit: str = "parts",
            min_interval: float = 5.0):
        self.label = label
        self.total = max(1, int(total))
        self.unit = unit
        self.min_interval = float(min_interval)
        self.started = time.time()
        self.last_print = 0.0

    def update(self, completed: int, part: str = "", force: bool = False):
        completed = max(0, min(int(completed), self.total))
        now = time.time()
        if (not force and completed < self.total and
                now - self.last_print < self.min_interval):
            return
        elapsed = now - self.started
        fraction = completed / self.total
        eta = (
            elapsed * (1 - fraction) / fraction
            if completed > 0 and fraction < 1 else 0.0)
        suffix = f" | part: {part}" if part else ""
        print(
            f"[progress] {self.label} {100*fraction:5.1f}% "
            f"({completed}/{self.total} {self.unit}){suffix} | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
            flush=True)
        self.last_print = now
