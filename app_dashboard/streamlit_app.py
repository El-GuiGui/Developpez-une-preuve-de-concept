# Pages : EDA, Comparaison, A propos

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageOps, ImageFilter

warnings.filterwarnings("ignore", category=FutureWarning)

from scripts.config import ensure_dirs, resolve_split_csv, CITYSCAPES_DIR, EXP_DIR
from scripts.preprocessing import (
    load_rgb, load_mask_labelids, remap_to_groups, colorize_groups, overlay,
    CATEGORY_NAMES, IGNORE_LABEL, N_CLASSES, PALETTE,
)
from scripts.augmentations import make_train_aug

TF_AVAILABLE = False
TORCH_AVAILABLE = False
try:
    import tensorflow as tf
    from scripts.losses_metrics import MeanIoUArgmax, dice_loss_sparse
    TF_AVAILABLE = True
except ImportError:
    pass
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass


# Config
st.set_page_config(
    page_title="Dashboard Segmentation — Cityscapes 8 classes",
    layout="wide",
    initial_sidebar_state="expanded",
)
ensure_dirs()

PLOTLY_COLORS = px.colors.qualitative.Safe

# Couleurs Plotly par classe, coherentes avec la palette des masques
CLASS_COLOR_MAP = {
    "void":         "rgb(0, 0, 0)",
    "flat":         "rgb(128, 64, 128)",
    "construction": "rgb(70, 70, 70)",
    "object":       "rgb(153, 153, 153)",
    "nature":       "rgb(107, 142, 35)",
    "sky":          "rgb(70, 130, 180)",
    "human":        "rgb(220, 20, 60)",
    "vehicle":      "rgb(0, 0, 142)",
}

# WCAG
PATTERN_SHAPES = ["/", "\\", "x", "+", "-", "|", ".", ""]
HIGH_CONTRAST_COLORS = [
    "#000000", "#D55E00", "#0072B2", "#CC79A7",
    "#009E73", "#56B4E9", "#E69F00", "#FFFFFF",
]
HIGH_CONTRAST_CLASS_MAP = {
    "void": "#000000", "flat": "#D55E00", "construction": "#0072B2",
    "object": "#CC79A7", "nature": "#009E73", "sky": "#56B4E9",
    "human": "#E69F00", "vehicle": "#FFFFFF",
}


# Data / paths

def resolve_path(row, col_abs, col_rel):
    if col_abs in row and isinstance(row[col_abs], str) and len(row[col_abs]) > 0:
        return row[col_abs]
    return f"{CITYSCAPES_DIR}/{row[col_rel]}"


EDA_CACHE_DIR = Path(__file__).resolve().parent / "eda_cache"
EDA_STATS_PATH = EDA_CACHE_DIR / "stats.json"
EDA_SAMPLES_DIR = EDA_CACHE_DIR / "samples"


@st.cache_data
def load_eda_cache():
    if not EDA_STATS_PATH.exists():
        return None
    return json.loads(EDA_STATS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_split_df():
    try:
        return pd.read_csv(resolve_split_csv())
    except (FileNotFoundError, RuntimeError):
        return None


def get_split_dfs(df):
    if df is None:
        return None, None, None
    return (
        df[df["split_final"] == "train"].copy(),
        df[df["split_final"] == "val"].copy(),
        df[df["split_final"] == "test"].copy(),
    )


# Experiments index

def _safe_read_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_float(x):
    try:
        return float(x) if x is not None else np.nan
    except (ValueError, TypeError):
        return np.nan


def find_first_existing(run_dir, candidates):
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    return None


def parse_run_name(run_name):
    s = run_name.lower()
    model_family, encoder, train_mode = None, None, None

    if "eomt" in s:
        model_family, encoder = "eomt", "dinov2"
    elif "segformer" in s:
        model_family, encoder = "segformer", "mit"
    elif "convnext" in s:
        model_family, encoder = "unet", "convnext_tiny"
    elif "resnet" in s:
        model_family, encoder = "unet", "resnet50"
    elif "vgg" in s:
        model_family, encoder = "unet", "vgg16"
    elif "unet" in s:
        model_family, encoder = "unet", "scratch"

    if "frozen" in s:
        train_mode = "frozen"
    elif any(x in s for x in ("finetune", "ft", "trainable")):
        train_mode = "finetune"
    elif encoder == "scratch":
        train_mode = "scratch"

    size_hw = None
    m = re.search(r"(\d{2,4})x(\d{2,4})", s)
    if m:
        size_hw = (int(m.group(1)), int(m.group(2)))

    epochs = None
    m = re.search(r"(?:_e|epoch)(\d+)", s)
    if m:
        epochs = int(m.group(1))

    batch = None
    m = re.search(r"(?:_b|batch)(\d+)", s)
    if m:
        batch = int(m.group(1))

    aug = None
    m = re.search(r"aug(\d+)", s)
    if m:
        aug = bool(int(m.group(1)))

    aug_repeats = 1
    m = re.search(r"rep(\d+)", s)
    if m:
        aug_repeats = int(m.group(1))

    loss_name = None
    if "ce_dice" in s or "cedice" in s:
        loss_name = "ce_dice"
    elif "ce" in s:
        loss_name = "ce"

    return dict(
        model_family=model_family, encoder=encoder, train_mode=train_mode,
        size_hw=size_hw, epochs=epochs, batch=batch,
        aug=aug, aug_repeats=aug_repeats, loss_name=loss_name,
    )


def extract_metrics_from_history(hist):
    if not hist or not isinstance(hist, dict):
        return {}

    def get_series(keys):
        for k in keys:
            if k in hist and isinstance(hist[k], list) and len(hist[k]) > 0:
                return np.array(hist[k], dtype=np.float32)
        return None

    out = {}
    loss = get_series(["loss"])
    vloss = get_series(["val_loss"])
    miou = get_series(["mIoU", "miou"])
    vmiou = get_series(["val_mIoU", "val_miou"])

    if loss is not None:
        out["last_loss"] = float(loss[-1])
    if vloss is not None:
        out["last_val_loss"] = float(vloss[-1])
        best_i = int(np.argmin(vloss))
        out["best_val_loss"] = float(vloss[best_i])
        out["best_val_loss_epoch"] = best_i + 1
    if miou is not None:
        out["last_mIoU"] = float(miou[-1])
    if vmiou is not None:
        out["last_val_mIoU"] = float(vmiou[-1])
        best_i = int(np.argmax(vmiou))
        out["best_val_mIoU"] = float(vmiou[best_i])
        out["best_val_mIoU_epoch"] = best_i + 1
        tail = vmiou[-min(5, len(vmiou)):]
        out["val_mIoU_tail_mean"] = float(np.mean(tail))
        out["val_mIoU_tail_std"] = float(np.std(tail))

    lr = get_series(["learning_rate", "lr"])
    if lr is not None:
        out["last_lr"] = float(lr[-1])

    n_epochs_ran = None
    if vloss is not None:
        n_epochs_ran = len(vloss)
    elif loss is not None:
        n_epochs_ran = len(loss)
    if n_epochs_ran is not None:
        out["n_epochs_ran"] = int(n_epochs_ran)

    return out


@st.cache_data
def load_runs_index(exp_dir):
    exp_dir = Path(exp_dir)
    rows = []
    if not exp_dir.exists():
        return pd.DataFrame(rows)

    for rd in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        run_name = rd.name
        summary = _safe_read_json(rd / "summary.json") or {}
        history = _safe_read_json(rd / "history.json") or {}
        parsed = parse_run_name(run_name)
        hist_metrics = extract_metrics_from_history(history)

        best_keras = find_first_existing(rd, ["best.keras", "best_model.keras", f"{run_name}.keras", "model.keras"])
        best_pt = find_first_existing(rd, ["best.pt", "best_model.pt", "model.pt", "best.pth", "best_model.pth", "model.pth"])
        pred_grid = find_first_existing(rd, ["pred_grid.png", "predictions.png", "overlay.png", "overlay_pred.png"])
        loss_png = find_first_existing(rd, ["loss.png"])
        miou_png = find_first_existing(rd, ["miou.png", "mIoU.png", "iou.png"])

        val_loss = safe_float(summary.get("val_loss"))
        val_mIoU = safe_float(summary.get("val_mIoU") or summary.get("val_miou"))
        test_loss = safe_float(summary.get("test_loss"))
        test_mIoU = safe_float(summary.get("test_mIoU") or summary.get("test_miou"))
        train_time_sec = safe_float(summary.get("train_time_sec"))
        test_mIoU_7 = safe_float(summary.get("test_mIoU_7_no_void"))
        test_mIoU_8 = safe_float(summary.get("test_mIoU_8_including_void"))
        infer_ms_per_img = safe_float(summary.get("infer_ms_per_img"))

        params = summary.get("params", {}) or {}
        if params:
            parsed["model_family"] = parsed["model_family"] or params.get("model")
            if params.get("size_hw"):
                parsed["size_hw"] = parsed["size_hw"] or tuple(params["size_hw"])
            parsed["batch"] = parsed["batch"] or params.get("batch")
            parsed["epochs"] = parsed["epochs"] or params.get("epochs")
            if parsed["aug"] is None:
                parsed["aug"] = params.get("aug")
            parsed["aug_repeats"] = parsed["aug_repeats"] or params.get("aug_repeats")
            parsed["loss_name"] = parsed["loss_name"] or params.get("loss_name")
            parsed["train_mode"] = parsed["train_mode"] or ("finetune" if params.get("trainable") else "frozen")
            parsed["encoder"] = parsed["encoder"] or params.get("encoder_preset") or params.get("encoder")

        best_model_path = str(best_keras) if best_keras else ""
        best_pt_path = str(best_pt) if best_pt else ""
        has_keras = bool(best_keras) and best_model_path.endswith(".keras")
        has_pytorch = bool(best_pt)

        size_mb = np.nan
        model_file = best_keras or best_pt
        if model_file:
            try:
                size_mb = float(model_file.stat().st_size / (1024 * 1024))
            except Exception:
                pass

        row = {
            "run_name": run_name, "run_dir": str(rd),
            **parsed,
            "val_loss": val_loss, "val_mIoU": val_mIoU,
            "test_loss": test_loss, "test_mIoU": test_mIoU,
            "train_time_sec": train_time_sec, "infer_ms_per_img": infer_ms_per_img,
            "test_mIoU_7_no_void": test_mIoU_7, "test_mIoU_8_including_void": test_mIoU_8,
            **hist_metrics,
            "best_model_path": best_model_path, "best_pt_path": best_pt_path,
            "has_keras": has_keras, "has_pytorch": has_pytorch, "size_mb": size_mb,
            "pred_grid_png": str(pred_grid) if pred_grid else "",
            "loss_png": str(loss_png) if loss_png else "",
            "miou_png": str(miou_png) if miou_png else "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Score principal : test_mIoU > best_val_mIoU > val_mIoU > EoMT mIoU_7
    df["score_main"] = df["test_mIoU"]
    mask_na = df["score_main"].isna()
    if "best_val_mIoU" in df.columns:
        df.loc[mask_na, "score_main"] = df.loc[mask_na, "best_val_mIoU"]
    mask_na = df["score_main"].isna()
    df.loc[mask_na, "score_main"] = df.loc[mask_na, "val_mIoU"]
    mask_na = df["score_main"].isna()
    df.loc[mask_na, "score_main"] = df.loc[mask_na, "test_mIoU_7_no_void"]

    return df.sort_values("score_main", ascending=False).reset_index(drop=True)


# EDA helpers

@st.cache_data
def compute_pixel_counts(df_split, n_samples, size_hw=(256, 256), seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df_split), size=min(n_samples, len(df_split)), replace=False)
    counts = np.zeros((N_CLASSES,), dtype=np.int64)
    H, W = size_hw
    for i in idx:
        row = df_split.iloc[int(i)]
        mask_path = resolve_path(row, "mask_path", "mask_rel")
        m = remap_to_groups(load_mask_labelids(mask_path)).resize((W, H), Image.NEAREST)
        a = np.array(m, dtype=np.uint8)
        counts += np.bincount(a[a != IGNORE_LABEL].flatten(), minlength=N_CLASSES)[:N_CLASSES]
    return pd.DataFrame({"class_id": range(N_CLASSES), "class_name": CATEGORY_NAMES, "pixels": counts})


@st.cache_data
def compute_presence_counts(df_split, n_samples, size_hw=(256, 256), seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df_split), size=min(n_samples, len(df_split)), replace=False)
    present = np.zeros((N_CLASSES,), dtype=np.int64)
    H, W = size_hw
    for i in idx:
        row = df_split.iloc[int(i)]
        mask_path = resolve_path(row, "mask_path", "mask_rel")
        m = remap_to_groups(load_mask_labelids(mask_path)).resize((W, H), Image.NEAREST)
        a = np.array(m, dtype=np.uint8)
        for c in np.unique(a[a != IGNORE_LABEL]):
            if 0 <= int(c) < N_CLASSES:
                present[int(c)] += 1
    return pd.DataFrame({"class_id": range(N_CLASSES), "class_name": CATEGORY_NAMES, "images_with_class": present})


def apply_albu_preview(img_pil, mask_pil, aug, size_hw=(256, 256), seed=0):
    img = img_pil.convert("RGB").resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    mask = mask_pil.resize((size_hw[1], size_hw[0]), Image.NEAREST)
    np.random.seed(seed)
    out = aug(image=np.array(img), mask=np.array(mask, dtype=np.uint8))
    return Image.fromarray(out["image"]), Image.fromarray(out["mask"].astype(np.uint8), mode="L")


# Sidebar components

def render_palette_legend():
    st.sidebar.markdown("### Légende des classes")
    for i, (name, color) in enumerate(zip(CATEGORY_NAMES, PALETTE)):
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_c = "#ffffff" if lum < 128 else "#000000"
        st.sidebar.markdown(
            f'<div style="background:{hex_c}; color:{text_c}; '
            f'padding:4px 10px; margin:2px 0; border-radius:4px; '
            f'font-weight:600; font-size:14px;">'
            f'{i} - {name}</div>',
            unsafe_allow_html=True,
        )


# WCAG

def inject_wcag_css():
    wcag_on = st.session_state.get("wcag_toggle", False)

    base_css = """
    a:focus-visible, button:focus-visible, input:focus-visible,
    select:focus-visible, textarea:focus-visible,
    [role="button"]:focus-visible, [tabindex]:focus-visible {
        outline: 3px solid #4da6ff !important;
        outline-offset: 2px !important;
    }
    """

    wcag_css = ""
    if wcag_on:
        wcag_css = """
    html, body, [class*="css"],
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stText, .stCaption, .stDataFrame,
    [data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    p, li, td, th, label, span, div {
        letter-spacing: 0.02em !important;
        word-spacing: 0.08em !important;
    }
    h1 { font-size: 2.4rem !important; }
    h2 { font-size: 2.0rem !important; }
    h3 { font-size: 1.6rem !important; }
    h1, h2, h3 { margin-bottom: 0.8em !important; }
    label, .stSelectbox label, .stSlider label,
    .stRadio label, .stCheckbox label {
        font-weight: 700 !important;
        font-size: 18px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    .stButton > button {
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 0.7rem 1.4rem !important;
        border-width: 2px !important;
    }
    a:focus-visible, button:focus-visible, input:focus-visible,
    select:focus-visible, textarea:focus-visible,
    [role="button"]:focus-visible, [tabindex]:focus-visible {
        outline: 4px solid #ff6600 !important;
        outline-offset: 3px !important;
        box-shadow: 0 0 0 6px rgba(255, 102, 0, 0.3) !important;
    }
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        font-size: 18px !important;
        min-height: 48px !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] { font-size: 16px !important; }
    .stDataFrame td, .stDataFrame th { font-size: 16px !important; padding: 8px 12px !important; }
    [data-testid="stSidebar"] { font-size: 17px !important; }
    [data-testid="stSidebar"] label { font-size: 17px !important; }
        """

    st.markdown(f"<style>{base_css}{wcag_css}</style>", unsafe_allow_html=True)

    if wcag_on:
        st.markdown(
            '<div style="background:#005fcc; color:white; padding:6px 16px; '
            'border-radius:4px; font-weight:600; font-size:14px; '
            'margin-bottom:10px; text-align:center;">'
            'Mode WCAG activé</div>',
            unsafe_allow_html=True,
        )


def _get_wcag_mode():
    return st.session_state.get("wcag_toggle", False)


# Chart helpers

def make_accessible_bar(df, x, y, title, text_col=None, color_col=None):
    wcag = _get_wcag_mode()
    is_class = (color_col == "class_name")

    if is_class:
        cmap = HIGH_CONTRAST_CLASS_MAP if wcag else CLASS_COLOR_MAP
        fig = px.bar(df, x=x, y=y, text=text_col or y, title=title, color=color_col, color_discrete_map=cmap)
    else:
        colors = HIGH_CONTRAST_COLORS if wcag else PLOTLY_COLORS
        fig = px.bar(df, x=x, y=y, text=text_col or y, title=title, color=color_col, color_discrete_sequence=colors)

    fig.update_traces(textposition="outside", textfont_size=12)
    if wcag:
        for i, trace in enumerate(fig.data):
            trace.marker.pattern.shape = PATTERN_SHAPES[i % len(PATTERN_SHAPES)]
            trace.marker.pattern.solidity = 0.6
            trace.marker.line.width = 1.5
            trace.marker.line.color = "white"

    fig.update_layout(
        xaxis_title=x, yaxis_title=y, xaxis_tickangle=-35,
        font=dict(size=13), title_font_size=16,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=14),
    )
    return fig


def make_accessible_scatter(df, x, y, title, color_col=None, symbol_col=None, hover_data=None):
    wcag = _get_wcag_mode()
    colors = HIGH_CONTRAST_COLORS if wcag else PLOTLY_COLORS
    fig = px.scatter(df, x=x, y=y, color=color_col, symbol=symbol_col,
                     hover_data=hover_data, title=title, color_discrete_sequence=colors)
    if wcag:
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color="white")))
    else:
        fig.update_traces(marker=dict(size=9))
    fig.update_layout(
        font=dict(size=13),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=14),
    )
    return fig


def render_chart_alt_text(description):
    with st.expander("Description du graphique (accessibilité)", expanded=False):
        st.write(description)


# Page EDA

def render_eda(train_df, val_df, test_df, size_hw, alpha):
    st.title("EDA — Cityscapes (8 classes)")

    cache = load_eda_cache()
    has_dataset = train_df is not None

    if cache:
        n_train, n_val, n_test = cache["train"]["n_images"], cache["val"]["n_images"], cache["test"]["n_images"]
    elif has_dataset:
        n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    else:
        st.error("Ni cache EDA ni dataset disponible. Lancez precompute_eda.py.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Train", n_train)
    c2.metric("Val", n_val)
    c3.metric("Test", n_test)

    # Exemples d'images
    st.subheader("Exemples d'images + masques (remap 8 classes)")
    split_choice = st.selectbox("Split", ["train", "val", "test"], index=2, key="eda_split")

    if cache and EDA_SAMPLES_DIR.exists():
        samples = [s for s in cache.get("samples", []) if s["split"] == split_choice]
        n_show = st.slider("Nombre d'exemples", 2, len(samples), min(6, len(samples)), 1, key="eda_nshow")
        cols = st.columns(3)
        for i in range(min(n_show, len(samples))):
            s = samples[i]
            img = Image.open(EDA_SAMPLES_DIR / s["img"]).convert("RGB")
            mask = Image.open(EDA_SAMPLES_DIR / s["mask"]).convert("L")
            m_rgb = colorize_groups(mask)
            ov = overlay(img, m_rgb, alpha=alpha)
            with cols[i % 3]:
                st.image(img, caption=f"Image #{i} ({split_choice})", use_container_width=True)
                st.image(m_rgb, caption=f"Masque #{i}", use_container_width=True)
                st.image(ov, caption=f"Overlay #{i}", use_container_width=True)
    elif has_dataset:
        df_split = {"train": train_df, "val": val_df, "test": test_df}[split_choice].reset_index(drop=True)
        n_show = st.slider("Nombre d'exemples", 2, 18, 6, 1, key="eda_nshow")
        cols = st.columns(3)
        for i in range(min(n_show, len(df_split))):
            row = df_split.iloc[i]
            img = load_rgb(resolve_path(row, "image_path", "image_rel")).resize((size_hw[1], size_hw[0]), Image.BILINEAR)
            m = remap_to_groups(load_mask_labelids(resolve_path(row, "mask_path", "mask_rel"))).resize((size_hw[1], size_hw[0]), Image.NEAREST)
            m_rgb = colorize_groups(m)
            ov = overlay(img, m_rgb, alpha=alpha)
            with cols[i % 3]:
                st.image(img, caption=f"Image #{i} ({split_choice})", use_container_width=True)
                st.image(m_rgb, caption=f"Masque #{i}", use_container_width=True)
                st.image(ov, caption=f"Overlay #{i}", use_container_width=True)

    # Comptages pixels par classe
    st.subheader("Comptages — pixels par classe")
    if cache:
        split_stats = cache[split_choice]
        counts_df = pd.DataFrame({"class_id": range(N_CLASSES), "class_name": CATEGORY_NAMES, "pixels": split_stats["pixel_counts"]})
        n_used = split_stats["n_images"]
        st.caption(f"Calculé sur la totalité du split ({n_used} images)")
    elif has_dataset:
        df_split = {"train": train_df, "val": val_df, "test": test_df}[split_choice].reset_index(drop=True)
        n_samples = st.slider("Échantillon (nb masques)", 50, 1500, 200, 50, key="eda_nsamples")
        seed = st.number_input("Seed", value=42, step=1, key="eda_seed")
        counts_df = compute_pixel_counts(df_split, n_samples=n_samples, size_hw=size_hw, seed=int(seed))
        n_used = min(n_samples, len(df_split))
    else:
        counts_df = None

    if counts_df is not None:
        fig_px = make_accessible_bar(counts_df, x="class_name", y="pixels",
                                     title=f"Pixels par classe (split={split_choice}, n={n_used})", color_col="class_name")
        st.plotly_chart(fig_px, use_container_width=True)
        top_class = counts_df.loc[counts_df["pixels"].idxmax(), "class_name"]
        render_chart_alt_text(f"Nombre de pixels par classe sur {n_used} masques du split {split_choice}. Classe dominante : {top_class}.")
        st.dataframe(counts_df, use_container_width=True, hide_index=True)

    # Présence des classes
    st.subheader("Présence des classes — images contenant la classe")
    if cache:
        split_stats = cache[split_choice]
        presence_df = pd.DataFrame({"class_id": range(N_CLASSES), "class_name": CATEGORY_NAMES, "images_with_class": split_stats["presence_counts"]})
    elif has_dataset:
        df_split = {"train": train_df, "val": val_df, "test": test_df}[split_choice].reset_index(drop=True)
        presence_df = compute_presence_counts(df_split, n_samples=n_samples, size_hw=size_hw, seed=int(seed))
    else:
        presence_df = None

    if presence_df is not None:
        fig_pr = make_accessible_bar(presence_df, x="class_name", y="images_with_class",
                                     title="Images contenant chaque classe", color_col="class_name")
        st.plotly_chart(fig_pr, use_container_width=True)
        top_p = presence_df.loc[presence_df["images_with_class"].idxmax(), "class_name"]
        render_chart_alt_text(f"Nombre d'images contenant chaque classe. Classe la plus fréquente : {top_p}.")
        st.dataframe(presence_df, use_container_width=True, hide_index=True)

    # Transformations
    st.subheader("Transformations — exemples")
    sample_img, sample_mask = None, None
    if cache and EDA_SAMPLES_DIR.exists():
        samples = [s for s in cache.get("samples", []) if s["split"] == split_choice]
        if samples:
            idx_t = st.slider("Index exemple", 0, len(samples) - 1, 0, 1, key="eda_idx_transform")
            sample_img = Image.open(EDA_SAMPLES_DIR / samples[idx_t]["img"]).convert("RGB")
            sample_mask = Image.open(EDA_SAMPLES_DIR / samples[idx_t]["mask"]).convert("L")
    elif has_dataset:
        df_split = {"train": train_df, "val": val_df, "test": test_df}[split_choice].reset_index(drop=True)
        idx_t = st.slider("Index exemple", 0, max(0, len(df_split) - 1), 0, 1, key="eda_idx_transform")
        row = df_split.iloc[int(idx_t)]
        sample_img = load_rgb(resolve_path(row, "image_path", "image_rel")).resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        sample_mask = remap_to_groups(load_mask_labelids(resolve_path(row, "mask_path", "mask_rel"))).resize((size_hw[1], size_hw[0]), Image.NEAREST)

    if sample_img is not None:
        t1, t2, t3 = st.columns(3)
        with t1:
            st.image(sample_img, caption="Original", use_container_width=True)
        with t2:
            st.image(ImageOps.equalize(sample_img), caption="Égalisation d'histogramme", use_container_width=True)
        with t3:
            st.image(sample_img.filter(ImageFilter.GaussianBlur(radius=2)), caption="Flou gaussien (r=2)", use_container_width=True)

        st.subheader("Augmentations (Albumentations)")
        show_aug = st.checkbox("Afficher les augmentations", value=True, key="eda_show_aug")
        if show_aug and sample_mask is not None:
            n_aug = st.slider("Nombre de variantes", 1, 9, 4, 1, key="eda_n_aug")
            aug = make_train_aug()
            grid_cols = st.columns(min(3, n_aug))
            for j in range(n_aug):
                img_aug, m_aug = apply_albu_preview(sample_img, sample_mask, aug, size_hw=size_hw, seed=100 + j)
                m_aug_rgb = colorize_groups(m_aug)
                ov = overlay(img_aug, m_aug_rgb, alpha=alpha)
                with grid_cols[j % min(3, n_aug)]:
                    st.image(img_aug, caption=f"Aug #{j}", use_container_width=True)
                    st.image(ov, caption=f"Overlay #{j}", use_container_width=True)


# Page Comparaison

def render_comparison(exp_df):
    st.title("Comparaison des runs")

    if exp_df is None or len(exp_df) == 0:
        st.error("Aucun run trouvé dans out/experiments/.")
        return

    df = exp_df.copy()

    # Filtres
    c1, c2, c3 = st.columns(3)
    with c1:
        fams = sorted([x for x in df["model_family"].dropna().unique() if x])
        sel_fams = st.multiselect("Famille modèle", fams, default=fams, key="cmp_fams")
    with c2:
        encs = sorted([x for x in df["encoder"].dropna().unique() if x])
        sel_encs = st.multiselect("Encodeur", encs, default=encs, key="cmp_encs")
    with c3:
        modes = sorted([x for x in df["train_mode"].dropna().unique() if x])
        sel_modes = st.multiselect("Mode entraînement", modes, default=modes, key="cmp_modes")

    if sel_fams:
        df = df[df["model_family"].isin(sel_fams)]
    if sel_encs:
        df = df[df["encoder"].isin(sel_encs)]
    if sel_modes:
        df = df[df["train_mode"].isin(sel_modes)]

    df = df.sort_values("score_main", ascending=False).reset_index(drop=True)

    # Tableau
    st.subheader("Tableau récapitulatif")
    default_cols = [
        "run_name", "model_family", "encoder", "train_mode",
        "size_hw", "epochs", "batch", "aug", "aug_repeats", "loss_name", "test_mIoU", "val_mIoU", "val_loss",
        "train_time_sec",
        "last_lr", "n_epochs_ran",
    ]
    available = [c for c in default_cols if c in df.columns]
    cols = st.multiselect("Colonnes", df.columns.tolist(), default=available, key="cmp_cols")
    view = df[cols].copy() if cols else df.copy()
    st.dataframe(view.fillna("-"), use_container_width=True, height=520)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger CSV", data=csv, file_name="comparison_runs.csv", mime="text/csv")

    # Graphiques
    st.subheader("Graphiques")
    c4, c5 = st.columns(2)

    with c4:
        metric = st.selectbox("Métrique (bar chart)", ["score_main", "best_val_mIoU", "best_val_loss", "train_time_sec"], index=0, key="cmp_metric")
        N = st.slider("Top N", 5, 30, 12, 1, key="cmp_topn")
        dd = df.copy()
        if metric in dd.columns:
            dd = dd[np.isfinite(dd[metric].astype(float))]
            asc = metric in ("best_val_loss", "train_time_sec")
            dd = dd.sort_values(metric, ascending=asc).head(int(N))
        if len(dd) > 0:
            fig = make_accessible_bar(dd, x="run_name", y=metric, title=f"Top {N} — {metric}", color_col="encoder")
            st.plotly_chart(fig, use_container_width=True)

    with c5:
        scatter_x = st.selectbox("Axe X", ["train_time_sec", "batch", "epochs", "size_mb"], index=0, key="cmp_x")
        scatter_y = st.selectbox("Axe Y", ["score_main", "best_val_mIoU", "val_mIoU"], index=0, key="cmp_y")
        dd = df.copy()
        if scatter_x in dd.columns and scatter_y in dd.columns:
            dd = dd[np.isfinite(dd[scatter_x].astype(float)) & np.isfinite(dd[scatter_y].astype(float))]
        if len(dd) > 0:
            fig2 = make_accessible_scatter(dd, x=scatter_x, y=scatter_y, title=f"{scatter_y} vs {scatter_x}",
                                           color_col="encoder", symbol_col="model_family",
                                           hover_data=["run_name", "train_mode", "loss_name"])
            st.plotly_chart(fig2, use_container_width=True)

    # Résumé par groupe
    st.subheader("Résumé par groupe")
    group_col = st.selectbox("Grouper par", ["encoder", "model_family", "train_mode", "loss_name"], index=0, key="cmp_group")
    metric2 = st.selectbox("Métrique", ["score_main", "best_val_mIoU", "val_mIoU"], index=0, key="cmp_group_metric")

    dd = df.copy()
    if metric2 in dd.columns:
        dd = dd[np.isfinite(dd[metric2].astype(float))]
    if len(dd) > 0 and group_col in dd.columns:
        g = dd.groupby(group_col)[metric2].agg(["count", "mean", "std", "max"]).sort_values("max", ascending=False)
        st.dataframe(g.round(4), use_container_width=True)

        wcag = _get_wcag_mode()
        box_colors = HIGH_CONTRAST_COLORS if wcag else PLOTLY_COLORS
        fig_box = px.box(dd, x=group_col, y=metric2, color=group_col,
                         title=f"Distribution {metric2} par {group_col}",
                         color_discrete_sequence=box_colors, points="all")
        fig_box.update_layout(font=dict(size=13), plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Détails d'un run
    st.subheader("Détails d'un run")
    run_choice = st.selectbox("Run", df["run_name"].tolist(), index=0, key="cmp_run")
    row = df[df["run_name"] == run_choice].iloc[0]
    run_dir = Path(row["run_dir"])

    history = _safe_read_json(run_dir / "history.json")
    if history:
        tab_loss, tab_miou = st.tabs(["Courbe Loss", "Courbe mIoU"])

        with tab_loss:
            if "loss" in history and "val_loss" in history:
                epochs_range = list(range(1, len(history["loss"]) + 1))
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=epochs_range, y=history["loss"], mode="lines", name="Train loss", line=dict(color="#2196F3", width=2)))
                fig_loss.add_trace(go.Scatter(x=epochs_range, y=history["val_loss"], mode="lines", name="Val loss", line=dict(color="#F44336", width=2)))
                fig_loss.update_layout(title=f"Loss — {run_choice}", xaxis_title="Epoch", yaxis_title="Loss",
                                       font=dict(size=13), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
                st.plotly_chart(fig_loss, use_container_width=True)

        with tab_miou:
            miou_key = "mIoU" if "mIoU" in history else "miou" if "miou" in history else None
            vmiou_key = "val_mIoU" if "val_mIoU" in history else "val_miou" if "val_miou" in history else None
            if miou_key and vmiou_key and miou_key in history and vmiou_key in history:
                epochs_range = list(range(1, len(history[miou_key]) + 1))
                fig_miou = go.Figure()
                fig_miou.add_trace(go.Scatter(x=epochs_range, y=history[miou_key], mode="lines", name="Train mIoU", line=dict(color="#4CAF50", width=2)))
                fig_miou.add_trace(go.Scatter(x=epochs_range, y=history[vmiou_key], mode="lines", name="Val mIoU", line=dict(color="#FF9800", width=2)))
                fig_miou.update_layout(title=f"mIoU — {run_choice}", xaxis_title="Epoch", yaxis_title="mIoU",
                                        font=dict(size=13), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
                st.plotly_chart(fig_miou, use_container_width=True)

    # Images sauvees du run
    cols_img = st.columns(3)
    for i, (key, label) in enumerate([("pred_grid_png", "Grille prédictions"), ("loss_png", "Loss"), ("miou_png", "mIoU")]):
        p = row.get(key, "")
        with cols_img[i]:
            if p and Path(p).exists():
                st.image(Image.open(p), caption=label, use_container_width=True)
            else:
                st.caption(f"{label} - non disponible")


# Page A propos

def render_about():
    st.title("À propos du projet")

    st.markdown("""
    ### Segmentation sémantique — Cityscapes

    Ce dashboard présente les résultats d'un projet de segmentation sémantique
    sur le dataset **Cityscapes** (images de caméras embarquées pour véhicules autonomes).

    **Objectif :** segmenter chaque pixel d'une image en **8 catégories principales** :
    void, flat, construction, object, nature, sky, human, vehicle.

    ---

    ### Modèles comparés

    | Modèle | Architecture | Encodeur | Framework |
    |--------|-------------|----------|-----------|
    | U-Net from scratch | U-Net | Aucun (from scratch) | Keras/TF |
    | U-Net + VGG16 | U-Net | VGG16 (ImageNet) | Keras/TF |
    | U-Net + ResNet50 | U-Net | ResNet50 (ImageNet) | Keras/TF |
    | U-Net + ConvNeXt Tiny | U-Net | ConvNeXt Tiny (ImageNet) | Keras/TF |
    | SegFormer (MiT-B0) | SegFormer | MiT-B0 (Cityscapes) | Keras/TF |
    | EoMT (DINOv2) | EoMT | DINOv2-Base | PyTorch |

    ---

    ### Méthodologie

    - **Preprocessing :** resize, normalisation [0, 1], remapping labelIds → 8 groupes
    - **Augmentations :** Albumentations (flip, brightness, blur, noise, rotation)
    - **Loss :** Sparse Categorical Cross-Entropy + Dice Loss (pondération 0.5)
    - **Optimiseur :** Adam (lr=1e-3 pour CNN, lr=6e-5 pour transformers)
    - **Callbacks :** ModelCheckpoint (best val mIoU), EarlyStopping, ReduceLROnPlateau
    - **Métrique principale :** mean IoU (mIoU) sur les 8 classes

    ---

    ### Accessibilité — Critères WCAG couverts

    Le mode **WCAG** (activable dans la sidebar) applique les critères suivants :

    | Critère WCAG | Description | Implémentation |
    |-------------|-------------|----------------|
    | **1.1.1** Non-text Content | Alternatives textuelles | Description sous chaque graphique + tableau de données |
    | **1.4.1** Use of Color | La couleur n'est pas le seul moyen | Hachures sur les barres + formes sur les scatter |
    | **1.4.3** Contrast (Minimum) | Ratio ≥ 4.5:1 | Palette haut contraste + valeurs numériques |
    | **1.4.4** Resize Text | Texte redimensionnable | Police 18px, titres agrandis |
    | **1.4.11** Non-text Contrast | Contraste éléments graphiques | Bordures sur barres et points |
    | **1.4.12** Text Spacing | Espacement du texte | Letter-spacing et word-spacing augmentés |
    | **2.4.7** Focus Visible | Indicateur de focus clavier | Outline orange 4px + halo (toujours actif) |
    """)

    st.subheader("Palette des 8 classes")
    palette_data = []
    for i, (name, color) in enumerate(zip(CATEGORY_NAMES, PALETTE)):
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        palette_data.append({"ID": i, "Classe": name, "Couleur": hex_c, "R": color[0], "G": color[1], "B": color[2]})
    st.dataframe(pd.DataFrame(palette_data), use_container_width=True, hide_index=True)

    cols = st.columns(8)
    for i, (name, color) in enumerate(zip(CATEGORY_NAMES, PALETTE)):
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_c = "#fff" if lum < 128 else "#000"
        with cols[i]:
            st.markdown(
                f'<div style="background:{hex_c}; color:{text_c}; text-align:center; padding:20px 5px; '
                f'border-radius:8px; font-weight:700; font-size:13px; min-height:80px; '
                f'display:flex; align-items:center; justify-content:center;">{name}</div>',
                unsafe_allow_html=True,
            )


# Main

df_all = load_split_df()
train_df, val_df, test_df = get_split_dfs(df_all)
exp_df = load_runs_index(str(EXP_DIR))

st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", ["EDA", "Comparaison", "À propos"], index=0, key="nav_page")

st.sidebar.markdown("---")
size_hw = (256, 256)
alpha = st.sidebar.slider("Alpha overlay", 0.0, 1.0, 0.45, 0.05, key="alpha_slider")

inject_wcag_css()
st.sidebar.checkbox("Mode accessible (WCAG)", value=False, key="wcag_toggle",
                     help="Police agrandie, hachures sur graphiques, palette haut contraste, focus renforcé.")

st.sidebar.markdown("---")
render_palette_legend()
st.sidebar.markdown("---")
st.sidebar.caption(f"Runs détectés: {len(exp_df)}")

if page == "EDA":
    render_eda(train_df, val_df, test_df, size_hw, alpha)
elif page == "Comparaison":
    render_comparison(exp_df)
else:
    render_about()
