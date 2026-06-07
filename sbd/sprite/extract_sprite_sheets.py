import csv
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

BUFFER_SIZE = 10240


@dataclass
class SpriteSheetDownloadInfo:
    cid: str
    location: str
    type: str


@dataclass
class SpriteSheet:
    idx: int
    start: str
    end: str
    duration: str
    cid: str
    location: str | None = None
    type: str | None = None

    def add_download_info(self, data: SpriteSheetDownloadInfo) -> None:
        self.location = data.location
        self.type = data.type

    @property
    def filename(self) -> str:
        return f"{self.idx}.{self.type.split('/')[-1]}"


def download_sprite_sheet(sprite_sheet: SpriteSheet, output_folderpath: Path | str):
    sprite_sheet_output_filepath = Path(output_folderpath) / sprite_sheet.filename
    urllib.request.urlretrieve(sprite_sheet.location, str(sprite_sheet_output_filepath))


def get_header_value(headers: str, header_name: str) -> Optional[str]:
    headers = re.sub(r"\r?\n[ \t]+", "", headers)
    pattern = rf"^{re.escape(header_name)}:\s*([^\r\n]+)"
    match = re.search(pattern, headers, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def get_content_type(headers: str) -> Optional[str]:
    content_type = get_header_value(headers, "Content-Type")
    if content_type:
        return content_type.split(";", 1)[0].strip().lower()
    return None


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
    segment = segment.removeprefix("\r").removeprefix("\n")
    segment = segment.removesuffix("\n").removesuffix("\r")
    return segment


def split_segment(segment: str) -> tuple[str, str]:
    if "\r\n\r\n" in segment:
        headers, body = segment.split("\r\n\r\n", 1)
    elif "\n\n" in segment:
        headers, body = segment.split("\n\n", 1)
    else:
        raise ValueError("Ill-formatted segment: Could not find header/body separator.")
    return headers, body


def extract_sprite_sheets_metadata(body: str) -> dict[str, SpriteSheet]:
    if not body.startswith("<!DOCTYPE html>"):
        raise ValueError("Ill-formatted segment #1: Could not find `<!DOCTYPE html>`.")
    sprite_sheets = {}
    pattern = re.compile(
        r"<figcaption>Slide\s+#(?P<idx>\d+):\s+(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+.\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})\s+\(duration:\s+(?P<duration>(?:\d+:)*\d+\.\d{3})\)<\/figcaption><img\s+src=\"cid:(?P<cid>[^\"]+)\">",
    )
    matches = re.finditer(pattern, body)
    for match in matches:
        sprite_sheet = SpriteSheet(**match.groupdict())
        sprite_sheets[sprite_sheet.cid] = sprite_sheet
    return sprite_sheets


def extract_sprite_sheet_download_info(headers: str) -> SpriteSheetDownloadInfo:
    cid = get_header_value(headers, "Content-ID").strip("<>")
    location = get_header_value(headers, "Content-Location")
    type = get_header_value(headers, "Content-Type")
    return SpriteSheetDownloadInfo(cid=cid, location=location, type=type)


def process_mhtml(filepath: Path | str, output_folderpath: Path | str):
    boundary = None
    segment_idx = 0
    mhtml_filepath = Path(filepath)
    temp_buffer_chunks: list[str] = []

    with mhtml_filepath.open("r", encoding="utf-8", errors="ignore", newline="") as file:
        while True:
            chunk = file.read(BUFFER_SIZE)
            if not chunk:
                break

            temp_buffer_chunks.append(chunk)

            if not boundary:
                joined_buffer = "".join(temp_buffer_chunks)
                boundary = read_boundary(joined_buffer)

            if boundary:
                joined_buffer = "".join(temp_buffer_chunks)
                segments = split_boundary_segments(joined_buffer, boundary)
                temp_buffer_chunks = [segments[-1]]

                for segment in segments[:-1]:  # Don't process last segment as it might be incomplete
                    if segment_idx == 0:
                        # Skip the very first segment of the file as it doesn't contain any relevant info now
                        segment_idx += 1
                        continue
                    headers, body = split_segment(strip_segment(segment))
                    if segment_idx == 1:
                        # Second segment contains the sprite sheets metadata
                        sprite_sheets = extract_sprite_sheets_metadata(body)
                    else:
                        # Following segments contain sprite sheets location / url
                        try:
                            dwl_info = extract_sprite_sheet_download_info(headers)
                        except Exception as e:
                            raise ValueError(f"Ill formatted segment #{segment_idx}: {e}")
                        sprite_sheets[dwl_info.cid].add_download_info(dwl_info)
                    segment_idx += 1

        if temp_buffer_chunks and boundary:
            remaining_part = strip_segment("".join(temp_buffer_chunks))
            if remaining_part and remaining_part != "--":
                try:
                    dwl_info = extract_sprite_sheet_download_info(headers)
                except Exception as e:
                    raise ValueError(f"Ill formatted segment #{segment_idx}: {e}")
                sprite_sheets[dwl_info.cid].add_download_info(dwl_info)

    output_folderpath = Path(output_folderpath)
    output_folderpath.mkdir(parents=True, exist_ok=True)
    download_sprite_sheet_fn = partial(download_sprite_sheet, output_folderpath=output_folderpath)
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(download_sprite_sheet_fn, sprite_sheets.values()))
    meta_filepath = output_folderpath / "meta.csv"
    with meta_filepath.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Idx", "Start", "End", "Duration", "Filepath"])
        for sprite_sheet in sprite_sheets.values():
            writer.writerow(
                [
                    sprite_sheet.idx,
                    sprite_sheet.start,
                    sprite_sheet.end,
                    sprite_sheet.duration,
                    sprite_sheet.filename,
                ]
            )
