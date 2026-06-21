"""Tests for sbd.subtitle module."""

import pytest
from datetime import timedelta
from pathlib import Path

from sbd.subtitle.parser import SubtitleParser, SRTParsingError
from sbd.subtitle.utils import remove_html_tags


def make_srt(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.srt"
    p.write_text(content, encoding="utf-8")
    return p


# ─── utils.py ─────────────────────────────────────────────────────────────────


class TestRemoveHtmlTags:
    def test_simple_bold_tag(self):
        assert remove_html_tags("<b>bold</b>") == "bold"

    def test_italic_tag(self):
        assert remove_html_tags("<i>italic</i>") == "italic"

    def test_font_tag_with_attributes(self):
        assert remove_html_tags('<font color="#fbff1c">text</font>') == "text"

    def test_nested_tags(self):
        assert remove_html_tags("<i><b>nested</b></i>") == "nested"

    def test_no_tags(self):
        assert remove_html_tags("plain text") == "plain text"

    def test_empty_string(self):
        assert remove_html_tags("") == ""

    def test_mixed_content(self):
        assert remove_html_tags("Hello <i>world</i>!") == "Hello world!"

    def test_malformed_closing_tag_removed(self):
        # e.g. </fount> instead of </font> — still matches <.*?>
        assert remove_html_tags("<font>text</fount>") == "text"

    def test_opened_not_closed_tag_removed(self):
        assert remove_html_tags("<font>text") == "text"

    def test_closeded_not_opened_tag_removed(self):
        assert remove_html_tags("text</i>") == "text"


# ─── parser.py ────────────────────────────────────────────────────────────────

SIMPLE_SRT = """\
1
00:02:16,612 --> 00:02:19,376
Senator, we're making
our final approach into Coruscant.

2
00:02:19,482 --> 00:02:21,609
Very good.

"""

SRT_WITH_HTML = """\
1
00:00:01,000 --> 00:00:03,000
Hello <b>world</b>!

2
00:00:04,000 --> 00:00:06,000
<i>Goodbye</i> <font color="#fff">cruel</font> world.

"""

SRT_WITH_COORDS = """\
1
00:03:20,476 --> 00:03:22,671 X1:117 X2:619 Y1:042 Y2:428
There was no danger at all.

"""

SRT_DOT_TIMESTAMPS = """\
1
00:02:16.612 --> 00:02:19.376
Dot-format timestamps.

"""

SRT_SHORT_TIMESTAMPS = """\
1
01:30,000 --> 01:32,500
Short MM:SS format.

"""

SRT_NO_TRAILING_NEWLINE = "1\n00:00:01,000 --> 00:00:03,000\nHello."

SRT_INVALID_INDEX = "not_a_number\n00:00:01,000 --> 00:00:03,000\nContent.\n\n"

SRT_INVALID_TIMESTAMPS = "1\nbad --> timestamps\nContent.\n\n"


class TestSubtitleParser:
    def test_subtitle_count(self, tmp_path):
        f = make_srt(tmp_path, SIMPLE_SRT)
        assert len(SubtitleParser.read(f).subtitles) == 2

    def test_subtitle_indices(self, tmp_path):
        f = make_srt(tmp_path, SIMPLE_SRT)
        subs = SubtitleParser.read(f).subtitles
        assert subs[0].idx == 1
        assert subs[1].idx == 2

    def test_timestamps_comma_format(self, tmp_path):
        f = make_srt(tmp_path, SIMPLE_SRT)
        sub = SubtitleParser.read(f).subtitles[0]
        assert sub.timestamp.start == timedelta(minutes=2, seconds=16, microseconds=612000)
        assert sub.timestamp.end == timedelta(minutes=2, seconds=19, microseconds=376000)

    def test_timestamps_dot_format(self, tmp_path):
        f = make_srt(tmp_path, SRT_DOT_TIMESTAMPS)
        sub = SubtitleParser.read(f).subtitles[0]
        assert sub.timestamp.start == timedelta(minutes=2, seconds=16, microseconds=612000)
        assert sub.timestamp.end == timedelta(minutes=2, seconds=19, microseconds=376000)

    def test_timestamps_short_mm_ss_format(self, tmp_path):
        f = make_srt(tmp_path, SRT_SHORT_TIMESTAMPS)
        sub = SubtitleParser.read(f).subtitles[0]
        assert sub.timestamp.start == timedelta(minutes=1, seconds=30)
        assert sub.timestamp.end == timedelta(minutes=1, seconds=32, microseconds=500000)

    def test_multiline_content_joined_with_space(self, tmp_path):
        f = make_srt(tmp_path, SIMPLE_SRT)
        sub = SubtitleParser.read(f).subtitles[0]
        assert sub.content == "Senator, we're making our final approach into Coruscant."

    def test_html_removed_by_default(self, tmp_path):
        f = make_srt(tmp_path, SRT_WITH_HTML)
        subs = SubtitleParser.read(f).subtitles
        assert subs[0].content == "Hello world!"
        assert subs[1].content == "Goodbye cruel world."

    def test_html_preserved_when_disabled(self, tmp_path):
        f = make_srt(tmp_path, SRT_WITH_HTML)
        subs = SubtitleParser.read(f, remove_html_tags=False).subtitles
        assert "<b>" in subs[0].content
        assert "<i>" in subs[1].content

    def test_no_coordinates_when_absent(self, tmp_path):
        f = make_srt(tmp_path, SIMPLE_SRT)
        assert SubtitleParser.read(f).subtitles[0].coordinates is None

    def test_coordinates_parsed(self, tmp_path):
        f = make_srt(tmp_path, SRT_WITH_COORDS)
        coords = SubtitleParser.read(f).subtitles[0].coordinates
        assert coords is not None
        assert coords.x1 == 117
        assert coords.x2 == 619
        assert coords.y1 == 42
        assert coords.y2 == 428

    def test_filepath_stored_on_subtitle(self, tmp_path):
        f = make_srt(tmp_path, SIMPLE_SRT)
        assert SubtitleParser.read(f).subtitles[0].filepath == f

    def test_empty_file_yields_no_subtitles(self, tmp_path):
        f = make_srt(tmp_path, "")
        assert SubtitleParser.read(f).subtitles == []

    def test_no_trailing_newline(self, tmp_path):
        f = make_srt(tmp_path, SRT_NO_TRAILING_NEWLINE)
        parser = SubtitleParser.read(f)
        assert len(parser.subtitles) == 1
        assert parser.subtitles[0].content == "Hello."

    def test_invalid_index_raises(self, tmp_path):
        f = make_srt(tmp_path, SRT_INVALID_INDEX)
        with pytest.raises(SRTParsingError):
            SubtitleParser.read(f)

    def test_invalid_timestamps_raises(self, tmp_path):
        f = make_srt(tmp_path, SRT_INVALID_TIMESTAMPS)
        with pytest.raises(SRTParsingError):
            SubtitleParser.read(f)
