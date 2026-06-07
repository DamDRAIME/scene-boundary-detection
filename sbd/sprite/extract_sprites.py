import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NewType

import cv2
import numpy as np

SpriteSheetImg = NewType("SpriteSheetImg", np.ndarray)
SpriteImg = NewType("SpriteImg", np.ndarray)


def split_from_grid(img: SpriteSheetImg, rows: int, cols: int) -> list[list[SpriteImg]]:
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


def timedelta_strptime(duration_str: str, format: str) -> timedelta:
    try:
        td = datetime.strptime(duration_str, format)
    except ValueError:
        td = datetime.strptime(duration_str, "%S.%f")
    return timedelta(hours=td.hour, minutes=td.minute, seconds=td.second, microseconds=td.microsecond)


def get_sprite_timestamp(
    sprite_local_idx: int, sprite_sheet_meta: dict[str, Any]
) -> tuple[datetime, datetime, datetime]:
    start_end_frmt = "%H:%M:%S,%f"
    ss_start = datetime.strptime(sprite_sheet_meta["Start"], start_end_frmt)
    ss_end = datetime.strptime(sprite_sheet_meta["End"], start_end_frmt)
    n_sprites = 9  # FIXME: Should come from ss_meta
    ss_duration = timedelta_strptime(sprite_sheet_meta["Duration"], "%M:%S.%f")
    s_duration = ss_duration / n_sprites
    s_start = ss_start + (s_duration * sprite_local_idx)
    s_end = ss_end if sprite_local_idx == n_sprites - 1 else s_start + s_duration
    return s_start, s_end, s_duration


def extract_sprites(src_folderpath: Path, dst_folderpath: Path, grid: tuple[int, int]) -> Path:
    src_folderpath, dst_folderpath = Path(src_folderpath), Path(dst_folderpath)
    dst_folderpath.mkdir(parents=True, exist_ok=True)

    sprite_sheet_meta_filepath = src_folderpath / "meta.csv"
    sprite_meta_filepath = dst_folderpath / "meta.csv"
    assert sprite_sheet_meta_filepath.exists(), f"Couldn't find the `meta.csv` file in folder {src_folderpath}"
    with (
        sprite_sheet_meta_filepath.open(mode="r", newline="", encoding="utf-8") as ss_meta_fh,
        sprite_meta_filepath.open("w", newline="") as s_meta_fh,
    ):
        ss_meta_reader = csv.DictReader(ss_meta_fh)
        s_meta_writer = csv.writer(s_meta_fh)
        s_meta_writer.writerow(["idx", "local_idx", "sprite_sheet_idx", "start", "end", "duration", "location"])
        global_idx = 0
        for sprite_sheet_meta in ss_meta_reader:
            sprite_sheet = cv2.imread(src_folderpath / sprite_sheet_meta["Filepath"])
            sprites = split_from_grid(sprite_sheet, *grid)  # FIXME: Should come from ss_meta
            for local_idx, sprite in enumerate(sprite for row in sprites for sprite in row):
                filename = f"sprite_{sprite_sheet_meta['Idx']}_{local_idx}.jpeg"
                time_frmt = "%M:%S.%f"
                start, end, duration = get_sprite_timestamp(local_idx, sprite_sheet_meta)
                cv2.imwrite(dst_folderpath / filename, sprite)
                s_meta_writer.writerow(
                    [
                        global_idx,
                        local_idx,
                        sprite_sheet_meta["Idx"],
                        start.strftime(time_frmt),
                        end.strftime(time_frmt),
                        str(duration),
                        filename,
                    ]
                )
                global_idx += 1

    return dst_folderpath
