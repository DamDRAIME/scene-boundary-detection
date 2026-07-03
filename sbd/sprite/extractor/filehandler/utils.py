from typing import Optional
import re

from sbd.sprite.exceptions import MHTMLParsingError
from sbd.sprite.extractor.filehandler.models import SpriteImg, SpriteSheetImg

_NO_SET = object()

# ############################### #
# Shape-related utility functions #
# ############################### #


def rescale_shape(shape: tuple[int, int], scale_ratio: float) -> tuple[int, int]:
    return tuple(map(lambda x: int(x * scale_ratio), shape))


def resolve_shape(
    src_shape: tuple[int, int], dst_shape: tuple[int | None, int | None], dst_scale_ratio: float = None
) -> tuple[int, int]:
    if dst_scale_ratio and any(dst_shape):
        raise ValueError("Pass either `dst_scale_ratio` or `dst_shape`, not both.")
    if dst_scale_ratio:
        return rescale_shape(src_shape, dst_scale_ratio)
    height, width = dst_shape
    if height and not width:
        return height, int(src_shape[1] * height / src_shape[0])
    if width and not height:
        return int(src_shape[0] * width / src_shape[1]), width
    if width and height:
        return height, width
    return src_shape


# ########################## #
# Cropping utility functions #
# ########################## #


def split_from_grid(img: SpriteSheetImg, grid_shape: tuple[int, int]) -> list[list[SpriteImg]]:
    rows, cols = grid_shape
    blocks = [[] for _ in range(cols)]
    img_h, img_w, _ = img.shape

    # Calculate individual block dimensions using floor division
    block_h = img_h // rows
    block_w = img_w // cols

    # Iterate through the grid coordinates
    for r in range(rows):
        for c in range(cols):
            # Compute exact cropping coordinates
            y1, y2 = r * block_h, (r + 1) * block_h
            x1, x2 = c * block_w, (c + 1) * block_w

            # Slice out the block
            blocks[r].append(img[y1:y2, x1:x2])

    return blocks


def cut_sprite_from_sprite_sheet(
    img: SpriteSheetImg, sprite_local_idx: int, grid_shape: tuple[int, int]
) -> list[list[SpriteImg]]:
    rows, cols = grid_shape
    # Calculate individual sprite dimensions using floor division
    img_h, img_w, _ = img.shape
    block_h = img_h // rows
    block_w = img_w // cols

    # Calculate sprite coordinates
    r = sprite_local_idx // cols
    c = sprite_local_idx % cols
    y1, y2 = r * block_h, (r + 1) * block_h
    x1, x2 = c * block_w, (c + 1) * block_w

    return img[y1:y2, x1:x2]


# ############################### #
# MHTML-related utility functions #
# ############################### #


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
