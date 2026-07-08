import re
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from sbd.common.models import Timestamps
from sbd.common.utils.detect_encoding import detect_encoding
from sbd.common.utils.timedelta import timedelta_parse
from sbd.exceptions import MHTMLParsingError, SpriteExtractionError
from sbd.sprite.extractor.base import SpriteFileHandler
from sbd.sprite.extractor.filehandler.models import (
    SourceMetadata,
    SpriteImg,
    SpriteSheet,
    SpriteSheetDownloadInfo,
    SpriteSheetImg,
)
from sbd.sprite.extractor.filehandler.utils import (
    get_header_value,
    read_boundary,
    split_boundary_segments,
    split_from_grid,
    split_segment,
    strip_segment,
)


class MHTMLFileHandler(SpriteFileHandler):
    spritesheet_metadata_pattern = re.compile(
        r"<figcaption>Slide\s+#(?P<idx>\d+):"  # Sprite index
        r"\s+(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+"  # Timestamp start
        r".?"  # Timestamps separator: Wildcard has it is hard to correctly decode
        r"\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})\s+"  # Timestamp end
        r"\(duration:\s+(?P<duration>(?:\d+:)*\d+\.\d{3})\)"  # Duration
        r"<\/figcaption><img\s+src=\"cid:(?P<cid>[^\"]+)\">"  # Content ID
    )

    def __init__(self, filepath: Path | str, grid_shape: tuple[int, int] = (3, 3), buffer_size: int = 10240):
        super().__init__(filepath)
        self.encoding = detect_encoding(self.filepath)
        self.grid_shape = grid_shape
        self.buffer_size = buffer_size
        self.sprite_sheets: list[SpriteSheet] = self._extract_sprite_sheets()
        self._src_meta = self.get_source_metadata()

    def iter_sprites(self, *args, **kwargs) -> Iterator[tuple[float, SpriteImg]]:
        for sprite_sheet in self.sprite_sheets[:-1]:
            yield from self._iter_sprites_from_sprite_sheet(sprite_sheet, self.grid_shape)

        # As the last sprite sheet might contain less sprite, it is handled independently from the others
        last_sprite_sheet = self.sprite_sheets[-1]
        grid_shape, n_sprites = self._infer_grid_shape_and_n_sprites(last_sprite_sheet)
        for (t, s), _ in zip(self._iter_sprites_from_sprite_sheet(last_sprite_sheet, grid_shape), range(n_sprites)):
            yield (t, s)

    def _iter_sprites_from_sprite_sheet(
        self, sprite_sheet: SpriteSheet, grid_shape: tuple[int, int]
    ) -> Iterator[tuple[float, SpriteImg]]:
        try:
            timestamp = sprite_sheet.timestamp.start
            ss_img = self._download_sprite_sheet(sprite_sheet)
            sprites_grid = split_from_grid(ss_img, grid_shape)
            sprites = [sprite for row in sprites_grid for sprite in row]
            for sprite in sprites:
                yield timestamp.total_seconds(), sprite
                timestamp += timedelta(seconds=1 / self.src_meta.fps)
        except Exception as e:
            raise SpriteExtractionError("An error occurred while extracting a sprite from a sprite sheet.") from e

    def get_source_metadata(self) -> SourceMetadata:
        sprite_sheet_shape = self._download_sprite_sheet(self.sprite_sheets[0]).shape[:2]
        sprite_shape = (sprite_sheet_shape[0] / self.grid_shape[0], sprite_sheet_shape[1] / self.grid_shape[1])
        n_sprites_per_sheet = self.grid_shape[0] * self.grid_shape[1]
        fps = 1 / (self.sprite_sheets[0].timestamp.duration.total_seconds() / n_sprites_per_sheet)
        n_sprites_last_sheet = int(self.sprite_sheets[-1].timestamp.duration.total_seconds() * fps)
        n_sprites = n_sprites_per_sheet * (len(self.sprite_sheets) - 1) + n_sprites_last_sheet
        duration = self.sprite_sheets[-1].timestamp.end
        return SourceMetadata(fps, duration, n_sprites, sprite_shape)

    def _extract_sprite_sheets(self) -> None:
        boundary = None
        segment_idx = 0
        buffer_chunks: list[str] = []

        with self.filepath.open("r", encoding=self.encoding, errors="ignore", newline="") as fh:
            while True:
                chunk = fh.read(self.buffer_size)
                if not chunk:
                    break

                buffer_chunks.append(chunk)

                if not boundary:
                    joined_buffer = "".join(buffer_chunks)
                    boundary = read_boundary(joined_buffer)

                if boundary:
                    joined_buffer = "".join(buffer_chunks)
                    segments = split_boundary_segments(joined_buffer, boundary)
                    buffer_chunks = [segments[-1]]

                    for segment in segments[:-1]:  # Don't process last segment as it might be incomplete
                        if segment_idx == 0:
                            # Skip the very first segment of the file as it doesn't contain any relevant info now
                            segment_idx += 1
                            continue
                        headers, body = split_segment(strip_segment(segment))
                        if segment_idx == 1:
                            # Second segment contains the sprite sheets metadata
                            sprite_sheets_map = self._extract_sprite_sheets_metadata(body)
                        else:
                            # Following segments contain sprite sheets location / url
                            try:
                                dwl_info = self._extract_sprite_sheet_download_info(headers)
                            except Exception as e:
                                raise MHTMLParsingError(f"Ill formatted segment #{segment_idx}: {e}")
                            sprite_sheets_map[dwl_info.cid].add_download_info(dwl_info)
                        segment_idx += 1

            if buffer_chunks and boundary:
                remaining_part = strip_segment("".join(buffer_chunks))
                if remaining_part and remaining_part != "--":
                    try:
                        dwl_info = self._extract_sprite_sheet_download_info(headers)
                    except Exception as e:
                        raise MHTMLParsingError(f"Ill formatted segment #{segment_idx}") from e
                    sprite_sheets_map[dwl_info.cid].add_download_info(dwl_info)
        return list(sprite_sheets_map.values())

    @staticmethod
    def _download_sprite_sheet(sprite_sheet: SpriteSheet) -> SpriteSheetImg:
        if not sprite_sheet.location:
            raise MHTMLParsingError(
                f"Could not download Sprite Sheet ({sprite_sheet.cid}) as its location wasn't found."
            )
        try:
            req = urllib.request.urlopen(sprite_sheet.location)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            raise SpriteExtractionError("An error occurred while downloading or converting the sprite sheet.") from e

    def _extract_sprite_sheets_metadata(self, body: str) -> dict[str, SpriteSheet]:
        if not body.startswith("<!DOCTYPE html>"):
            raise MHTMLParsingError("Ill-formatted segment #1: Could not find `<!DOCTYPE html>`.")

        sprite_sheets = {}
        matches = re.finditer(self.spritesheet_metadata_pattern, body)
        for match in matches:
            timestamp = Timestamps(timedelta_parse(match.group("start")), timedelta_parse(match.group("end")))
            sprite_sheet = SpriteSheet(match.group("idx"), timestamp, match.group("cid"), self.grid_shape)
            sprite_sheets[sprite_sheet.cid] = sprite_sheet
        if not sprite_sheets:
            raise MHTMLParsingError("Ill-formatted body: Could not find sprite sheets' metadata. Check regex pattern.")
        return sprite_sheets

    def _extract_sprite_sheet_download_info(self, headers: str) -> SpriteSheetDownloadInfo:
        cid = get_header_value(headers, "Content-ID").strip("<>")
        location = get_header_value(headers, "Content-Location")
        type = get_header_value(headers, "Content-Type", default=None)
        return SpriteSheetDownloadInfo(cid=cid, location=location, type=type)

    def _infer_grid_shape_and_n_sprites(self, sprite_sheet: SpriteSheet) -> tuple[tuple[int, int], int]:
        n_sprites = int(sprite_sheet.timestamp.duration.total_seconds() * self.src_meta.fps)
        if n_sprites <= self.grid_shape[1]:
            return (1, n_sprites), n_sprites
        return (1 + ((n_sprites - 1) // self.grid_shape[0]), self.grid_shape[1]), n_sprites
