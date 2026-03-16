import io
import os
import re
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Config
st.set_page_config(page_title="Cityscapes Segmentation", layout="wide")
API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")

N_CLASSES = 8
IGNORE_LABEL = 255

CLASS_NAMES = [
    "void",
    "flat",
    "construction",
    "object",
    "nature",
    "sky",
    "human",
    "vehicle",
]

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


# Helpers
def cityscapes_key(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]

    m = re.match(r"(.+)_leftImg8bit$", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.match(
        r"(.+)_gtFine_(labelIds|labelTrainIds|color|instanceIds)$",
        stem,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1)

    return stem


# labelIds ->
# void: 0..6 (+255 ignore)
# flat: 7..10
# construction: 11..16
# object: 17..20
# nature: 21..22
# sky: 23
# human: 24..25
# vehicle: 26..33

LABELIDS_TO_GROUP = np.full(256, 0, dtype=np.uint8)
for i in [7, 8, 9, 10]:
    LABELIDS_TO_GROUP[i] = 1
for i in [11, 12, 13, 14, 15, 16]:
    LABELIDS_TO_GROUP[i] = 2
for i in [17, 18, 19, 20]:
    LABELIDS_TO_GROUP[i] = 3
for i in [21, 22]:
    LABELIDS_TO_GROUP[i] = 4
LABELIDS_TO_GROUP[23] = 5
for i in [24, 25]:
    LABELIDS_TO_GROUP[i] = 6
for i in [26, 27, 28, 29, 30, 31, 32, 33]:
    LABELIDS_TO_GROUP[i] = 7


def remap_labelids_to_groups(mask_labelids: np.ndarray) -> np.ndarray:
    m = mask_labelids.astype(np.int32)
    out = np.zeros_like(m, dtype=np.uint8)

    ignore = m == IGNORE_LABEL
    m_clip = np.clip(m, 0, 255).astype(np.uint8)
    out = LABELIDS_TO_GROUP[m_clip]
    out[ignore] = IGNORE_LABEL
    return out.astype(np.uint8)


def mask_to_vis_gray(mask_0_7: np.ndarray) -> Image.Image:
    m = mask_0_7.copy()
    m = np.where(m == IGNORE_LABEL, 0, m).astype(np.float32)
    m = (m * (255.0 / (N_CLASSES - 1))).astype(np.uint8)
    return Image.fromarray(m, mode="L")


def colorize_groups(mask_0_7: np.ndarray) -> Image.Image:
    m = mask_0_7.astype(np.int64)
    m = np.where(m == IGNORE_LABEL, 0, m)
    m = np.clip(m, 0, len(PALETTE) - 1)
    rgb = PALETTE[m]
    return Image.fromarray(rgb, mode="RGB")


def overlay(
    rgb_img: Image.Image, mask_rgb: Image.Image, alpha: float = 0.45
) -> Image.Image:
    rgb = rgb_img.convert("RGBA")
    m = mask_rgb.convert("RGBA")
    m.putalpha(int(255 * alpha))
    return Image.alpha_composite(rgb, m).convert("RGB")


def decode_png_bytes_to_np(png_bytes: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(png_bytes))
    if im.mode not in ("L", "I;16"):
        im = im.convert("L")
    arr = np.array(im)
    return arr


def pred_mask_from_api(mask_png: bytes) -> np.ndarray:
    arr = decode_png_bytes_to_np(mask_png)

    # case A: already 0..7
    if arr.max() <= 7:
        return arr.astype(np.uint8)

    # case B: "visual" 0..255 of 0..7
    # map back by rounding
    if arr.max() <= 255:
        est = np.rint(arr.astype(np.float32) * (N_CLASSES - 1) / 255.0)
        est = np.clip(est, 0, N_CLASSES - 1).astype(np.uint8)
        return est

    # fallback
    return np.clip(arr, 0, 7).astype(np.uint8)


def gt_mask_from_upload(gt_bytes: bytes, target_hw: tuple[int, int]) -> np.ndarray:
    """Load GT png and return groups 0..7 in target size (H,W)."""
    im = Image.open(io.BytesIO(gt_bytes))

    # Cityscapes masks are often single channel; enforce "L"
    if im.mode not in ("L", "I;16"):
        im = im.convert("L")

    # resize to prediction size
    H, W = target_hw
    im = im.resize((W, H), Image.NEAREST)
    arr = np.array(im)

    # If it's already grouped 0..7 -> keep
    if arr.max() <= 7:
        return arr.astype(np.uint8)

    # Else it's probably labelIds -> remap
    return remap_labelids_to_groups(arr)


def confusion_matrix(
    gt: np.ndarray, pred: np.ndarray, num_classes: int = N_CLASSES
) -> np.ndarray:
    """gt/pred: (H,W) uint8, gt may contain 255 ignore."""
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch GT {gt.shape} vs Pred {pred.shape}")

    valid = gt != IGNORE_LABEL
    gt_v = gt[valid].astype(np.int64)
    pr_v = pred[valid].astype(np.int64)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = num_classes * gt_v + pr_v
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    cm[:] = binc.reshape((num_classes, num_classes))
    return cm


def iou_from_cm(cm: np.ndarray) -> tuple[np.ndarray, float]:
    ious = np.zeros((cm.shape[0],), dtype=np.float64)
    for c in range(cm.shape[0]):
        inter = cm[c, c]
        union = cm[c, :].sum() + cm[:, c].sum() - inter
        ious[c] = (inter / union) if union > 0 else 0.0
    miou = float(np.mean(ious))
    return ious, miou


def pixel_stats(mask: np.ndarray) -> pd.DataFrame:
    """mask: 0..7 or 255 ignore; returns table per class."""
    m = mask.copy()
    m = m[m != IGNORE_LABEL]
    total = int(m.size) if m.size else 1
    rows = []
    for cid in range(N_CLASSES):
        px = int((m == cid).sum())
        rows.append(
            {
                "class_id": cid,
                "class_name": CLASS_NAMES[cid],
                "pixels": px,
                "pct": 100.0 * px / total,
            }
        )
    df = (
        pd.DataFrame(rows).sort_values("pixels", ascending=False).reset_index(drop=True)
    )
    return df


# UI
st.title("Segmentation Cityscapes")

left, right = st.columns([1.15, 1.0])

with left:
    st.subheader("Entrée (RGB)")
    rgb_up = st.file_uploader(
        "Charge une image RGB (png/jpg)",
        type=["png", "jpg", "jpeg"],
        key="rgb_uploader",
    )

    rgb_img = None
    rgb_bytes = None
    rgb_name = None

    if rgb_up is not None:
        rgb_name = rgb_up.name
        rgb_bytes = rgb_up.getvalue()
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        st.image(rgb_img, caption="Image originale", width="stretch")

with right:
    st.subheader("Sorties API")
    if rgb_bytes is not None:

        def post_png(endpoint: str) -> bytes:
            r = requests.post(
                f"{API_BASE}{endpoint}",
                files={"file": (rgb_name or "image.png", rgb_bytes, "image/png")},
                timeout=180,
            )
            r.raise_for_status()
            return r.content

        try:
            t0 = time.time()
            mask_png = post_png("/predict/mask")
            color_png = post_png("/predict/mask_color")
            overlay_png = post_png("/predict/overlay")
            dt = time.time() - t0

            pred_mask = pred_mask_from_api(mask_png)

            mask_vis = mask_to_vis_gray(pred_mask)
            color_img = Image.open(io.BytesIO(color_png)).convert("RGB")
            overlay_img = Image.open(io.BytesIO(overlay_png)).convert("RGB")

            a, b, c = st.columns(3)
            with a:
                st.image(mask_vis, caption="Mask brut (visu)", width="stretch")
                st.download_button(
                    "Télécharger mask brut",
                    data=mask_png,
                    file_name="mask_pred.png",
                    mime="image/png",
                )
            with b:
                st.image(color_img, caption="Mask colorisé", width="stretch")
                st.download_button(
                    "Télécharger mask colorisé",
                    data=color_png,
                    file_name="mask_pred_color.png",
                    mime="image/png",
                )
            with c:
                st.image(overlay_img, caption="Overlay (Pred)", width="stretch")
                st.download_button(
                    "Télécharger overlay",
                    data=overlay_png,
                    file_name="overlay_pred.png",
                    mime="image/png",
                )

            st.caption(f"Temps total requêtes: {dt:.2f}s — API: {API_BASE}")

            with st.expander("Stats prédiction (classes / pixels)", expanded=True):
                st.dataframe(pixel_stats(pred_mask), width="stretch", hide_index=True)

        except Exception as e:
            st.error(f"Erreur API: {e}")
            st.caption(f"API: {API_BASE}")
            pred_mask = None

st.divider()


st.subheader("Masque Vérité terrain ")
st.caption("Charge un masque GT (gtFine_labelIds).")

gt_up = st.file_uploader(
    "Charge un masque GT",
    type=["png"],
    key="gt_uploader",
)

if gt_up is not None and rgb_up is not None:
    try:
        gt_name = gt_up.name
        gt_bytes = gt_up.getvalue()

        # Need prediction mask to evaluate
        if "pred_mask" not in locals() or pred_mask is None:
            st.warning("Aucune prédiction disponible (API KO ou pas d'image RGB).")
        else:
            H, W = pred_mask.shape[:2]
            gt_mask = gt_mask_from_upload(gt_bytes, target_hw=(H, W))

            # filename matching (warning only)
            k_rgb = cityscapes_key(rgb_name or "")
            k_gt = cityscapes_key(gt_name or "")
            if k_rgb and k_gt and (k_rgb != k_gt):
                st.warning(
                    f"⚠️ Les noms ne semblent pas correspondre.\n"
                    f"RGB key: {k_rgb} | GT key: {k_gt}\n"
                    "Je calcule quand même l'IoU (mais ce sera incohérent si ce n’est pas la même scène)."
                )

            # Compute IoU
            cm = confusion_matrix(gt_mask, pred_mask, num_classes=N_CLASSES)
            ious, miou = iou_from_cm(cm)

            # UI
            colA, colB = st.columns([1.0, 1.0])
            with colA:
                st.image(
                    mask_to_vis_gray(gt_mask), caption="GT (visu gris)", width="stretch"
                )

            with colB:
                st.markdown("### Évaluation (GT vs Pred)")
                st.metric("mIoU (GT vs Pred)", f"{miou:.4f}")

                rows = []
                for cid in range(N_CLASSES):
                    rows.append(
                        {
                            "class_id": cid,
                            "class_name": CLASS_NAMES[cid],
                            "iou": float(ious[cid]),
                        }
                    )
                df_iou = (
                    pd.DataFrame(rows)
                    .sort_values("iou", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(df_iou, width="stretch", hide_index=True)

    except Exception as e:
        st.error(f"Erreur GT/IoU: {e}")
elif gt_up is not None and rgb_up is None:
    st.info("Uploader d'abord une image RGB, puis un GT correspondant.")
