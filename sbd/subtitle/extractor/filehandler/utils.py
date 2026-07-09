import re

HTML_TAG = re.compile("<.*?>")
ASS_OVERRIDE_BLOCK = re.compile(r"\{.*?\}")


def remove_html_tags(x: str) -> str:
    return re.sub(HTML_TAG, "", x)


def remove_ass_override_blocks(x: str) -> str:
    return re.sub(ASS_OVERRIDE_BLOCK, "", x)
