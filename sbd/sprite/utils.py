from sbd.sprite.models import SpriteImg, SpriteSheetImg


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
