"""Tests for sbd.sprite module."""

import pytest
from datetime import timedelta

import numpy as np

from sbd.shared.models import Timestamps
from sbd.sprite.exceptions import MHTMLParsingError
from sbd.sprite.models import Sprite, SpriteSheet, SpriteSheetDownloadInfo
from sbd.sprite.utils import cut_sprite_from_sprite_sheet, split_from_grid
from sbd.sprite import mhtml


# ─── helpers ──────────────────────────────────────────────────────────────────


def make_sheet(idx=1, start=0, end=9, grid=(3, 3), cid="abc@cid", location=None, type=None):
    ts = Timestamps(timedelta(seconds=start), timedelta(seconds=end))
    return SpriteSheet(idx=idx, timestamp=ts, cid=cid, grid_shape=grid, location=location, type=type)


def colored_grid(rows, cols, block_h=10, block_w=10):
    """Build an image where each cell has a distinct constant grayscale value."""
    img = np.zeros((rows * block_h, cols * block_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            val = r * cols + c
            img[r * block_h : (r + 1) * block_h, c * block_w : (c + 1) * block_w] = val
    return img


# ─── models.py ────────────────────────────────────────────────────────────────


class TestSpriteSheet:
    def test_n_sprites(self):
        assert make_sheet(grid=(3, 3)).n_sprites == 9

    def test_n_sprites_non_square(self):
        assert make_sheet(grid=(2, 5)).n_sprites == 10

    def test_filename_jpeg(self):
        assert make_sheet(idx=3, type="image/jpeg").filename == "sprite_sheet_3.jpeg"

    def test_filename_png(self):
        assert make_sheet(idx=1, type="image/png").filename == "sprite_sheet_1.png"

    def test_add_download_info(self):
        ss = make_sheet()
        ss.add_download_info(SpriteSheetDownloadInfo(cid="abc@cid", location="http://example.com/img.jpg", type="image/jpeg"))
        assert ss.location == "http://example.com/img.jpg"
        assert ss.type == "image/jpeg"


class TestSprite:
    def _sheet(self):
        return make_sheet(start=0, end=9, grid=(3, 3), type="image/jpeg")

    def test_first_sprite_start(self):
        sprite = Sprite(idx=1, local_idx=0, sprite_sheet=self._sheet())
        assert sprite.timestamp.start == timedelta(seconds=0)

    def test_first_sprite_end(self):
        sprite = Sprite(idx=1, local_idx=0, sprite_sheet=self._sheet())
        assert sprite.timestamp.end == timedelta(seconds=1)

    def test_middle_sprite_start(self):
        sprite = Sprite(idx=4, local_idx=3, sprite_sheet=self._sheet())
        assert sprite.timestamp.start == timedelta(seconds=3)

    def test_last_sprite_end_equals_sheet_end(self):
        sprite = Sprite(idx=9, local_idx=8, sprite_sheet=self._sheet())
        assert sprite.timestamp.end == timedelta(seconds=9)

    def test_filename(self):
        sprite = Sprite(idx=5, local_idx=3, sprite_sheet=make_sheet(idx=2, type="image/jpeg"))
        assert sprite.filename == "sprite_sheet_2_sprite_3_5.jpeg"


# ─── utils.py ─────────────────────────────────────────────────────────────────


class TestSplitFromGrid:
    def test_block_count(self):
        blocks = split_from_grid(colored_grid(3, 3), (3, 3))
        assert len(blocks) == 3
        assert all(len(row) == 3 for row in blocks)

    def test_block_shape(self):
        blocks = split_from_grid(colored_grid(3, 3, block_h=20, block_w=30), (3, 3))
        assert blocks[0][0].shape == (20, 30, 3)

    def test_block_content(self):
        blocks = split_from_grid(colored_grid(2, 2), (2, 2))
        assert np.all(blocks[0][0] == 0)
        assert np.all(blocks[0][1] == 1)
        assert np.all(blocks[1][0] == 2)
        assert np.all(blocks[1][1] == 3)

    def test_non_square_grid(self):
        # split_from_grid pre-allocates `cols` outer lists; only the first `rows` are populated
        blocks = split_from_grid(colored_grid(2, 4), (2, 4))
        assert len(blocks) == 4
        assert len(blocks[0]) == 4  # row 0: all 4 columns filled
        assert len(blocks[1]) == 4  # row 1: all 4 columns filled
        assert blocks[2] == [] and blocks[3] == []  # trailing lists unused


class TestCutSpriteFromSpriteSheet:
    def test_shape(self):
        sprite = cut_sprite_from_sprite_sheet(colored_grid(3, 3, block_h=20, block_w=30), 0, (3, 3))
        assert sprite.shape == (20, 30, 3)

    def test_first_sprite(self):
        sprite = cut_sprite_from_sprite_sheet(colored_grid(2, 2), 0, (2, 2))
        assert np.all(sprite == 0)

    def test_last_sprite(self):
        sprite = cut_sprite_from_sprite_sheet(colored_grid(2, 2), 3, (2, 2))
        assert np.all(sprite == 3)

    def test_middle_sprite(self):
        # local_idx=4 in a 2x3 grid → row=1, col=1 → value = 1*3 + 1 = 4
        sprite = cut_sprite_from_sprite_sheet(colored_grid(2, 3), 4, (2, 3))
        assert np.all(sprite == 4)


# ─── mhtml.py ─────────────────────────────────────────────────────────────────


class TestGetHeaderValue:
    def test_found(self):
        headers = "Content-Type: text/html\r\nContent-ID: <abc123>"
        assert mhtml.get_header_value(headers, "Content-ID") == "<abc123>"

    def test_case_insensitive(self):
        assert mhtml.get_header_value("content-type: text/html", "Content-Type") == "text/html"

    def test_missing_with_none_default(self):
        assert mhtml.get_header_value("", "X-Missing", default=None) is None

    def test_missing_with_string_default(self):
        assert mhtml.get_header_value("", "X-Missing", default="fallback") == "fallback"

    def test_missing_raises(self):
        with pytest.raises(MHTMLParsingError):
            mhtml.get_header_value("", "X-Missing")

    def test_folded_header_unfolded(self):
        # RFC 2822 folding: \r\n followed by whitespace is collapsed to nothing
        headers = "Content-Location: http://example.com\r\n /path"
        assert mhtml.get_header_value(headers, "Content-Location") == "http://example.com/path"


class TestGetContentType:
    def test_strips_parameters(self):
        assert mhtml.get_content_type("Content-Type: text/html; charset=utf-8") == "text/html"

    def test_lowercased(self):
        assert mhtml.get_content_type("Content-Type: Image/JPEG") == "image/jpeg"

    def test_missing_with_default(self):
        assert mhtml.get_content_type("", default=None) is None

    def test_missing_raises(self):
        with pytest.raises(MHTMLParsingError):
            mhtml.get_content_type("")


class TestReadBoundary:
    def test_found(self):
        preamble = 'Content-Type: multipart/related; boundary="----=_Part_0_12345"'
        assert mhtml.read_boundary(preamble) == "----=_Part_0_12345"

    def test_not_found(self):
        assert mhtml.read_boundary("Content-Type: text/html") is None

    def test_case_insensitive(self):
        assert mhtml.read_boundary('Content-Type: multipart/related; BOUNDARY="myboundary"') == "myboundary"


class TestSplitBoundarySegments:
    def test_splits_on_boundary(self):
        parts = mhtml.split_boundary_segments("preamble\r\n--B\r\npart1\r\n--B\r\npart2", "B")
        assert len(parts) == 3
        assert parts[0] == "preamble"
        assert parts[1] == "part1"
        assert parts[2] == "part2"

    def test_closing_boundary_counted(self):
        parts = mhtml.split_boundary_segments("--B\r\npart1\r\n--B--", "B")
        assert len(parts) == 3


class TestStripSegment:
    def test_strips_leading_lf(self):
        assert mhtml.strip_segment("\ndata") == "data"

    def test_strips_leading_cr(self):
        assert mhtml.strip_segment("\rdata") == "data"

    def test_strips_trailing_lf(self):
        assert mhtml.strip_segment("data\n") == "data"

    def test_strips_trailing_cr(self):
        assert mhtml.strip_segment("data\r") == "data"

    def test_strips_crlf(self):
        assert mhtml.strip_segment("\r\ndata\r\n") == "data"

    def test_no_stripping_needed(self):
        assert mhtml.strip_segment("data") == "data"


class TestSplitSegment:
    def test_crlf_separator(self):
        headers, body = mhtml.split_segment("Header: value\r\n\r\nbody content")
        assert headers == "Header: value"
        assert body == "body content"

    def test_lf_separator(self):
        headers, body = mhtml.split_segment("Header: value\n\nbody content")
        assert headers == "Header: value"
        assert body == "body content"

    def test_no_separator_raises(self):
        with pytest.raises(MHTMLParsingError):
            mhtml.split_segment("no separator here")

    def test_crlf_preferred_over_lf(self):
        headers, body = mhtml.split_segment("H: v\r\n\r\nbody\n\nnot split")
        assert body == "body\n\nnot split"
