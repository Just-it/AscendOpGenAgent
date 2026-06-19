import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PerfResult:
    """Performance measurement result."""
    name: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    n_runs: int
    n_warmup: int = 0
    cv_pct: float = 0.0
    n_outliers: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "n_runs": self.n_runs,
            "n_warmup": self.n_warmup,
            "cv_pct": round(self.cv_pct, 2),
            "n_outliers": self.n_outliers,
        }


def _measure_npu_event(fn: Callable, n_warmup: int, n_runs: int,
                       name: str) -> PerfResult:
    """Measure using torch_npu.npu.Event for device-side timing."""
    import torch
    import torch_npu

    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.npu.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    return _build_result_robust(name, times, n_warmup)


def _measure_perf_counter(fn: Callable, n_warmup: int, n_runs: int,
                          sync_fn: Optional[Callable],
                          name: str) -> PerfResult:
    """Measure using time.perf_counter (fallback for CPU or non-NPU)."""
    for _ in range(n_warmup):
        fn()
        if sync_fn:
            sync_fn()

    times = []
    for _ in range(n_runs):
        if sync_fn:
            sync_fn()
        t0 = time.perf_counter()
        fn()
        if sync_fn:
            sync_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return _build_result_robust(name, times, n_warmup)


def _build_result(name: str, times: list, n_warmup: int = 0) -> PerfResult:
    """Build PerfResult from a list of timing measurements in ms."""
    mean_val = statistics.mean(times)
    std_val = statistics.stdev(times) if len(times) > 1 else 0.0
    return PerfResult(
        name=name,
        mean_ms=mean_val,
        median_ms=statistics.median(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=std_val,
        n_runs=len(times),
        n_warmup=n_warmup,
        cv_pct=(std_val / mean_val * 100) if mean_val > 0 else 0.0,
        n_outliers=0,
    )


def _build_result_robust(name: str, times: list, n_warmup: int = 0) -> PerfResult:
    """Build PerfResult with IQR outlier removal."""
    if len(times) < 4:
        r = _build_result(name, times, n_warmup)
        return r

    sorted_t = sorted(times)
    q1 = sorted_t[len(sorted_t) // 4]
    q3 = sorted_t[3 * len(sorted_t) // 4]
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    clean = [t for t in times if lower <= t <= upper]
    n_outliers = len(times) - len(clean)
    if len(clean) < 3:
        clean = times
        n_outliers = 0

    mean_val = statistics.mean(clean)
    std_val = statistics.stdev(clean) if len(clean) > 1 else 0.0

    return PerfResult(
        name=name,
        mean_ms=mean_val,
        median_ms=statistics.median(clean),
        min_ms=min(clean),
        max_ms=max(clean),
        std_ms=std_val,
        n_runs=len(clean),
        n_warmup=n_warmup,
        cv_pct=(std_val / mean_val * 100) if mean_val > 0 else 0.0,
        n_outliers=n_outliers,
    )


def _has_npu() -> bool:
    """Check if NPU device is available."""
    try:
        import torch
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


def measure_time(fn: Callable, n_warmup: int = 10, n_runs: int = 100,
                 sync_fn: Optional[Callable] = None,
                 name: str = "",
                 use_npu_event: Optional[bool] = None) -> PerfResult:
    """Measure execution time of a function.

    Args:
        fn: Function to measure (no args, returns anything).
        n_warmup: Number of warmup runs.
        n_runs: Number of timed runs.
        sync_fn: Synchronization function (e.g., torch.npu.synchronize).
                 Only used with perf_counter fallback.
        name: Label for the result.
        use_npu_event: Force NPU event timing (True), perf_counter (False),
                       or auto-detect (None, default).
    """
    if use_npu_event is None:
        use_npu_event = _has_npu()

    if use_npu_event:
        return _measure_npu_event(fn, n_warmup, n_runs, name)
    return _measure_perf_counter(fn, n_warmup, n_runs, sync_fn, name)
