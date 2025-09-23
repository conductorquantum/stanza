def seconds_to_ns(seconds: float) -> int:
    """Convert seconds to nanoseconds, rounded to the nearest integer.

    Args:
        seconds (float): Time in seconds

    Returns:
        int: Time in nanoseconds
    """
    return int(round(seconds * 1e9))
