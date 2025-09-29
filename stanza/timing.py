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
