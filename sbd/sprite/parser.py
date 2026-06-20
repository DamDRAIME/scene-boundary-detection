from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
import csv
from pathlib import Path
import re
import urllib.request

import cv2
import numpy as np

from sbd.shared.models import Timestamps
from sbd.shared.utils.counter import Counter
from sbd.sprite import mhtml
from sbd.sprite.exceptions import MHTMLParsingError
from sbd.sprite.models import Sprite, SpriteSheet, SpriteSheetDownloadInfo, SpriteSheetImg
from sbd.shared.utils.detect_encoding import detect_encoding
from sbd.shared.utils.timedelta import timedelta_parse
from sbd.sprite.utils import cut_sprite_from_sprite_sheet, split_from_grid


class SpriteParser:
    spritesheet_metadata_pattern = re.compile(
        r"<figcaption>Slide\s+#(?P<idx>\d+):"  # Sprite index
        r"\s+(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+"  # Timestamp start
        r"."  # Timestamps separator: Wildcard has it is hard to correctly decode
        r"\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})\s+"  # Timestamp end
        r"\(duration:\s+(?P<duration>(?:\d+:)*\d+\.\d{3})\)"  # Duration
        r"<\/figcaption><img\s+src=\"cid:(?P<cid>[^\"]+)\">"  # Content ID
    )

    def __init__(self, filepath: Path | str, grid_shape: tuple[int, int] = (3, 3), buffer_size: int = 10240):
        self.filepath = Path(filepath)
        self.encoding = detect_encoding(self.filepath)
        self.grid_shape = grid_shape
        self.buffer_size = buffer_size
        self.sprite_sheets: list[SpriteSheet] = []
        self.sprites: list[Sprite] = []

    def parse(self):
        self.extract_sprite_sheets()
        self.extract_sprites()

    @classmethod
    def read(
        cls, filepath: Path | str, grid_shape: tuple[int, int] = (3, 3), buffer_size: int = 10240
    ) -> "SpriteParser":
        _self = cls(filepath, grid_shape, buffer_size)
        _self.parse()
        return _self

    def download(self, output_folderpath: Path | str) -> Path:
        output_folderpath = Path(output_folderpath)
        output_ss = output_folderpath / "sprite_sheets"
        output_ss.mkdir(parents=True, exist_ok=True)
        ss_meta_filepath = output_folderpath / "sprite_sheets_meta.csv"
        output_s = output_folderpath / "sprites"
        output_s.mkdir(parents=True, exist_ok=True)
        s_meta_filepath = output_folderpath / "sprites_meta.csv"

        sprite_sheet_to_sprite = defaultdict(list)
        for sprite in self.sprites:
            sprite_sheet_to_sprite[sprite.sprite_sheet.idx].append(sprite)

        with ss_meta_filepath.open("w", newline="") as ss_meta_fh, s_meta_filepath.open("w", newline="") as s_meta_fh:
            ss_writer = csv.writer(ss_meta_fh)
            ss_writer.writerow(["idx", "start", "end", "duration", "location"])
            s_meta_writer = csv.writer(s_meta_fh)
            s_meta_writer.writerow(["idx", "local_idx", "sprite_sheet_idx", "start", "end", "duration", "location"])

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_ss = {executor.submit(self._download_sprite_sheet, ss): ss for ss in self.sprite_sheets}
                for future in as_completed(future_to_ss):
                    sprite_sheet = future_to_ss[future]
                    sprite_sheet_img = future.result()
                    cv2.imwrite(output_ss / sprite_sheet.filename, sprite_sheet_img)
                    ss_writer.writerow(
                        [
                            sprite_sheet.idx,
                            str(sprite_sheet.timestamp.start),
                            str(sprite_sheet.timestamp.end),
                            str(sprite_sheet.timestamp.duration),
                            output_ss / sprite_sheet.filename,
                        ]
                    )
                    for sprite in sprite_sheet_to_sprite[sprite_sheet.idx]:
                        sprite_img = cut_sprite_from_sprite_sheet(
                            sprite_sheet_img, sprite.local_idx, sprite_sheet.grid_shape
                        )
                        cv2.imwrite(output_s / sprite.filename, sprite_img)
                        s_meta_writer.writerow(
                            [
                                sprite.idx,
                                sprite.local_idx,
                                sprite.sprite_sheet.idx,
                                str(sprite.timestamp.start),
                                str(sprite.timestamp.end),
                                str(sprite.timestamp.duration),
                                output_s / sprite.filename,
                            ]
                        )

    def extract_sprite_sheets(self) -> None:
        self.sprite_sheets: list[SpriteSheet] = []  # Reset for the method to be idempotent
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
                    boundary = mhtml.read_boundary(joined_buffer)

                if boundary:
                    joined_buffer = "".join(buffer_chunks)
                    segments = mhtml.split_boundary_segments(joined_buffer, boundary)
                    buffer_chunks = [segments[-1]]

                    for segment in segments[:-1]:  # Don't process last segment as it might be incomplete
                        if segment_idx == 0:
                            # Skip the very first segment of the file as it doesn't contain any relevant info now
                            segment_idx += 1
                            continue
                        headers, body = mhtml.split_segment(mhtml.strip_segment(segment))
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
                remaining_part = mhtml.strip_segment("".join(buffer_chunks))
                if remaining_part and remaining_part != "--":
                    try:
                        dwl_info = self._extract_sprite_sheet_download_info(headers)
                    except Exception as e:
                        raise MHTMLParsingError(f"Ill formatted segment #{segment_idx}: {e}")
                    sprite_sheets_map[dwl_info.cid].add_download_info(dwl_info)
        self.sprite_sheets = list(sprite_sheets_map.values())

    def extract_sprites(self) -> None:
        self.sprites: list[Sprite] = []  # Reset for the method to be idempotent
        global_idx_generator = Counter()
        for sprite_sheet in self.sprite_sheets:
            for local_idx in range(sprite_sheet.n_sprites):
                self.sprites.append(Sprite(global_idx_generator.next(), local_idx, sprite_sheet))

    @staticmethod
    def download_sprite_sheet(sprite_sheet: SpriteSheet, output_folderpath: Path | str) -> Path:
        sprite_sheet_output_filepath = Path(output_folderpath) / sprite_sheet.filename
        urllib.request.urlretrieve(sprite_sheet.location, str(sprite_sheet_output_filepath))
        return sprite_sheet_output_filepath

    @staticmethod
    def download_sprite(sprite: Sprite, output_folderpath: Path | str) -> Path:
        req = urllib.request.urlopen(sprite.sprite_sheet.location)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        sprite_sheet_img = cv2.imdecode(arr, -1)
        sprite_img = cut_sprite_from_sprite_sheet(sprite_sheet_img, sprite.local_idx, sprite.sprite_sheet.grid_shape)
        sprite_output_filepath = Path(output_folderpath) / sprite.filename
        cv2.imwrite(sprite_output_filepath, sprite_img)
        return sprite_output_filepath

    @staticmethod
    def _download_sprite_sheet(sprite_sheet: SpriteSheet) -> SpriteSheetImg:
        req = urllib.request.urlopen(sprite_sheet.location)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(arr, -1)

    def _extract_sprite_sheets_metadata(self, body: str) -> dict[str, SpriteSheet]:
        if not body.startswith("<!DOCTYPE html>"):
            raise MHTMLParsingError("Ill-formatted segment #1: Could not find `<!DOCTYPE html>`.")

        sprite_sheets = {}
        matches = re.finditer(self.spritesheet_metadata_pattern, body)
        for match in matches:
            timestamp = Timestamps(timedelta_parse(match.group("start")), timedelta_parse(match.group("end")))
            sprite_sheet = SpriteSheet(match.group("idx"), timestamp, match.group("cid"), self.grid_shape)
            sprite_sheets[sprite_sheet.cid] = sprite_sheet
        return sprite_sheets

    def _extract_sprite_sheet_download_info(self, headers: str) -> SpriteSheetDownloadInfo:
        cid = mhtml.get_header_value(headers, "Content-ID").strip("<>")
        location = mhtml.get_header_value(headers, "Content-Location")
        type = mhtml.get_header_value(headers, "Content-Type", default=None)
        return SpriteSheetDownloadInfo(cid=cid, location=location, type=type)
