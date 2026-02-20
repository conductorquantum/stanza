import datetime


def seconds_to_ns(seconds: float) -> int:
    """Convert seconds to nanoseconds, rounded to the nearest integer.

    Args:
        seconds (float): Time in seconds

    Returns:
        int: Time in nanoseconds
    """
    return int(round(seconds * 1e9))


def to_epoch(timestamp: float | int | datetime.datetime) -> float:
    """Convert a timestamp to epoch time.

    Args:
        timestamp (float): Timestamp

    Returns:
        float: Epoch time
    """
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    return timestamp.timestamp()


QUA_CYCLE_NS = 4  # 1 QUA clock cycle = 4 nanoseconds


def ns_to_cycles(ns: int) -> int:
    """Convert nanoseconds to QUA clock cycles. Raises ValueError if not aligned to 4 ns."""
    if ns % QUA_CYCLE_NS != 0:
        raise ValueError(
            f"{ns} ns is not aligned to {QUA_CYCLE_NS} ns QUA cycle boundary"
        )
    return ns // QUA_CYCLE_NS


def cycles_to_ns(cycles: int) -> int:
    """Convert QUA clock cycles to nanoseconds."""
    return cycles * QUA_CYCLE_NS


def ns_to_samples(ns: int, sample_rate_hz: float = 1e9) -> int:
    """Convert nanoseconds to sample count at given sample rate."""
    return int(ns * sample_rate_hz / 1e9)


def samples_to_ns(samples: int, sample_rate_hz: float = 1e9) -> int:
    """Convert sample count to nanoseconds at given sample rate."""
    return int(samples * 1e9 / sample_rate_hz)
