import re
from typing import Optional

from sbd.sprite.exceptions import MHTMLParsingError

_NO_SET = object()


def get_header_value(headers: str, header_name: str, default: Optional[str] = _NO_SET) -> Optional[str]:
    headers = re.sub(r"\r?\n[ \t]+", "", headers)
    pattern = rf"^{re.escape(header_name)}:\s*([^\r\n]+)"
    match = re.search(pattern, headers, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    elif default is not _NO_SET:
        return default
    raise MHTMLParsingError(f"Header {header_name} not found.")


def get_content_type(headers: str, default: Optional[str] = _NO_SET) -> Optional[str]:
    content_type = get_header_value(headers, "Content-Type", default)
    if content_type == default:
        return content_type
    return content_type.split(";", 1)[0].strip().lower()


def read_boundary(temp_buffer: str) -> Optional[str]:
    boundary_pattern = re.compile(r'boundary="([^"]+)"', re.IGNORECASE)
    boundary_match = boundary_pattern.search(temp_buffer)
    if boundary_match:
        boundary = boundary_match.group(1).strip()
        if boundary:
            return boundary
    return None


def split_boundary_segments(buffer: str, boundary: str) -> list[str]:
    """Split only on MIME boundary delimiter lines."""
    boundary_segment_delimiter_pattern = re.compile(rf"(?:^|\r?\n)--{re.escape(boundary)}(?:--)?[ \t]*(?:\r?\n|$)")
    return boundary_segment_delimiter_pattern.split(buffer)


def strip_segment(segment: str) -> str:
    """Remove MIME boundary separator line breaks without trimming payload bytes."""
    return segment.removeprefix("\r").removeprefix("\n").removesuffix("\n").removesuffix("\r")


def split_segment(segment: str) -> tuple[str, str]:
    if "\r\n\r\n" in segment:
        headers, body = segment.split("\r\n\r\n", 1)
    elif "\n\n" in segment:
        headers, body = segment.split("\n\n", 1)
    else:
        raise MHTMLParsingError("Ill-formatted segment: Could not find header/body separator.")
    return headers, body
