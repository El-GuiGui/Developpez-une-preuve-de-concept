import numpy as np
from PIL import Image

PALETTE = np.array(
    [
        [0, 0, 0],  # void
        [128, 64, 128],  # flat (road/sidewalk)
        [70, 70, 70],  # construction (building/wall)
        [153, 153, 153],  # object (pole/traffic sign)
        [107, 142, 35],  # nature (vegetation)
        [70, 130, 180],  # sky
        [220, 20, 60],  # human (person/rider)
        [0, 0, 142],  # vehicle (car/bus/truck)
    ],
    dtype=np.uint8,
)


def colorize_groups(mask_2d: np.ndarray) -> Image.Image:
    mask = mask_2d.astype(np.int64)
    mask = np.clip(mask, 0, len(PALETTE) - 1)
    rgb = PALETTE[mask]
    return Image.fromarray(rgb, mode="RGB")


def overlay(
    rgb_img: Image.Image, mask_rgb: Image.Image, alpha: float = 0.45
) -> Image.Image:
    rgb = rgb_img.convert("RGBA")
    m = mask_rgb.convert("RGBA")
    m.putalpha(int(255 * alpha))
    return Image.alpha_composite(rgb, m).convert("RGB")
