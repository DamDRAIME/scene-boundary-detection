import re

HTML_TAG = re.compile("<.*?>")


def remove_html_tags(x: str) -> str:
    return re.sub(HTML_TAG, "", x)
