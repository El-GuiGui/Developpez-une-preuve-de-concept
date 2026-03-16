from pathlib import Path
from typing import Tuple
import os
import tempfile

import numpy as np
import tensorflow as tf
import requests
from PIL import Image

from scripts.viz import colorize_groups, overlay
from scripts.losses_metrics import MeanIoUArgmax, dice_loss_sparse

DEFAULT_SIZE_HW: Tuple[int, int] = (256, 256)
DEFAULT_MODEL_PATH = Path("models") / "best_model.keras"

HF_MODEL_URL = "https://huggingface.co/GuiLL-L/my-best-model-unet-for-proj8-cityscapes/resolve/main/best_model.keras?download=true"

_MODEL = None


def _download_file_atomic(url: str, dst_path: Path, timeout: int = 600) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        with tempfile.NamedTemporaryFile(
            delete=False, dir=str(dst_path.parent), suffix=".tmp"
        ) as tmp:
            tmp_name = tmp.name
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    tmp.write(chunk)

    os.replace(tmp_name, str(dst_path))


def ensure_model_present(
    model_path: Path = DEFAULT_MODEL_PATH,
    force: bool = False,
    min_bytes: int = 10_000_000,
) -> None:
    if force or (not model_path.exists()) or (model_path.stat().st_size < min_bytes):
        _download_file_atomic(HF_MODEL_URL, model_path)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    global _MODEL
    if _MODEL is None:
        model_path = Path(model_path)

        ensure_model_present(model_path)

        _MODEL = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                "MeanIoUArgmax": MeanIoUArgmax,
                "dice_loss_sparse": dice_loss_sparse,
            },
            compile=False,
        )
    return _MODEL


def preprocess_pil(
    img: Image.Image, size_hw: Tuple[int, int] = DEFAULT_SIZE_HW
) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def predict_from_pil(
    img: Image.Image,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    size_hw: Tuple[int, int] = DEFAULT_SIZE_HW,
    alpha: float = 0.45,
):
    model = load_model(model_path)

    x = preprocess_pil(img, size_hw=size_hw)
    pred = model.predict(x[None, ...], verbose=0)[0]

    mask = np.argmax(pred, axis=-1).astype(np.uint8)
    resized_rgb = Image.fromarray((x * 255).astype(np.uint8), mode="RGB")

    mask_color = colorize_groups(mask)
    overlay_img = overlay(resized_rgb, mask_color, alpha=alpha)

    return mask, mask_color, overlay_img, resized_rgb
