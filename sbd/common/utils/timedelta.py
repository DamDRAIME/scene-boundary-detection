from datetime import datetime, timedelta


def timedelta_strptime(time_string: str, format: str) -> timedelta:
    dt = datetime.strptime(time_string, format)
    return timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)


def timedelta_parse(time_string: str) -> timedelta:
    expected_formats = ["%H:%M:%S", "%H:%M:%S,%f", "%H:%M:%S.%f", "%M:%S,%f", "%M:%S.%f", "%S.%f"]
    for format in expected_formats:
        try:
            return timedelta_strptime(time_string, format)
        except ValueError:
            continue
    raise ValueError()


def convert_to_seconds(timestamp: float | timedelta) -> float:
    """Convert a timestamp to seconds."""
    if isinstance(timestamp, timedelta):
        return timestamp.total_seconds()
    return timestamp
