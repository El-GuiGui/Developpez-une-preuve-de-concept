import numpy as np
from PIL import Image

CATEGORY_NAMES = [
    "void",
    "flat",
    "construction",
    "object",
    "nature",
    "sky",
    "human",
    "vehicle",
]

IGNORE_LABEL = 255
N_CLASSES = 8

LABELID_TO_CATEGORY = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 4,
    22: 4,
    23: 5,
    24: 6,
    25: 6,
    26: 7,
    27: 7,
    28: 7,
    29: 7,
    30: 7,
    31: 7,
    32: 7,
    33: 7,
}


def build_lut():
    lut = np.zeros((256,), dtype=np.uint8)
    for k, v in LABELID_TO_CATEGORY.items():
        lut[k] = v
    lut[IGNORE_LABEL] = IGNORE_LABEL
    return lut


LUT = build_lut()

PALETTE = np.array(
    [
        [0, 0, 0],  # void
        [128, 64, 128],  # flat
        [70, 70, 70],  # construction
        [153, 153, 153],  # object
        [107, 142, 35],  # nature
        [70, 130, 180],  # sky
        [220, 20, 60],  # human
        [0, 0, 142],  # vehicle
    ],
    dtype=np.uint8,
)


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_labelids(path: str) -> Image.Image:
    return Image.open(path)


def remap_to_groups(mask_pil: Image.Image) -> Image.Image:
    a = np.array(mask_pil, dtype=np.uint16)
    g = LUT[a]
    return Image.fromarray(g.astype(np.uint8), mode="L")


def colorize_groups(mask_groups: Image.Image) -> Image.Image:
    g = np.array(mask_groups, dtype=np.uint8)
    h, w = g.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = g != IGNORE_LABEL
    rgb[valid] = PALETTE[g[valid]]
    rgb[~valid] = np.array([255, 0, 255], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay(
    img_rgb: Image.Image, mask_rgb: Image.Image, alpha: float = 0.45
) -> Image.Image:
    return Image.blend(
        img_rgb.convert("RGBA"),
        mask_rgb.convert("RGBA"),
        alpha,
    ).convert("RGB")
