import re
from datetime import datetime, timedelta

HTML_TAG = re.compile("<.*?>")


def remove_html_tags(x: str) -> str:
    return re.sub(HTML_TAG, "", x)


def timedelta_strptime(time_string: str, format: str) -> timedelta:
    dt = datetime.strptime(time_string, format)
    return timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
