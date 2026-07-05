from datetime import timedelta


def convert_to_seconds(timestamp: float | timedelta) -> float:
    """Convert a timestamp to seconds."""
    if isinstance(timestamp, timedelta):
        return timestamp.total_seconds()
    return timestamp
