from __future__ import annotations

from datetime import datetime


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log_info(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def log_stage_start(stage: str, *, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    log_info(f"{stage}: starting{suffix}")


def log_stage_done(stage: str, *, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    log_info(f"{stage}: done{suffix}")


def log_skip(stage: str, *, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    log_info(f"{stage}: skipped (cached){suffix}")


class ProgressTracker:
    """Emit periodic progress lines for long loops."""

    def __init__(self, total: int, label: str, *, step_pct: int = 5) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.step_pct = max(1, min(step_pct, 100))
        self.current = 0
        self._next_pct = self.step_pct

    def advance(self, n: int = 1) -> None:
        self.current = min(self.current + n, self.total)
        pct = int(100 * self.current / self.total)
        if self.current >= self.total or pct >= self._next_pct:
            log_info(f"  {self.label}: {self.current}/{self.total} ({pct}%)")
            self._next_pct = ((pct // self.step_pct) + 1) * self.step_pct
