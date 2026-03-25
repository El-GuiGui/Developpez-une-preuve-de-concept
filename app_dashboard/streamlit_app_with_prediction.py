"""
Dashboard Streamlit de segmentation sémantique Cityscapes (8 classes).

Pages :
  1. EDA          — exploration du dataset (images, masques, comptages, transformations)
  2. Prédiction   — inférence sur image test ou uploadée (Keras + PyTorch/EoMT)
  3. Comparaison  — tableau + graphiques interactifs de tous les runs
  4. À propos     — légende palette, méthodologie, infos projet
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ajout du répertoire racine au path
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

# Imports projet
from scripts.config import ensure_dirs, resolve_split_csv, CITYSCAPES_DIR, EXP_DIR
from scripts.preprocessing import (
    load_rgb,
    load_mask_labelids,
    remap_to_groups,
    colorize_groups,
    overlay,
    CATEGORY_NAMES,
    IGNORE_LABEL,
    N_CLASSES,
    PALETTE,
)
from scripts.augmentations import make_train_aug

# Imports conditionnels (Keras / PyTorch)
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


# CONFIGURATION PAGE

st.set_page_config(
    page_title="Dashboard Segmentation — Cityscapes 8 classes",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_dirs()

# Palette colorblind-friendly pour graphiques encodeurs/modèles
PLOTLY_COLORS = px.colors.qualitative.Safe

# Couleurs par CLASSE
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


# HELPERS – DONNÉES / CHEMINS

def resolve_path(row, col_abs: str, col_rel: str) -> str:
    """Résout le chemin absolu ou relatif d'une image/masque."""
    if col_abs in row and isinstance(row[col_abs], str) and len(row[col_abs]) > 0:
        return row[col_abs]
    return f"{CITYSCAPES_DIR}/{row[col_rel]}"


@st.cache_data
def load_split_df() -> pd.DataFrame:
    csv_path = resolve_split_csv()
    return pd.read_csv(csv_path)


def get_split_dfs(df: pd.DataFrame):
    return (
        df[df["split_final"] == "train"].copy(),
        df[df["split_final"] == "val"].copy(),
        df[df["split_final"] == "test"].copy(),
    )



# HELPERS – EXPERIMENTS INDEX

def _safe_read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_float(x):
    try:
        return float(x) if x is not None else np.nan
    except (ValueError, TypeError):
        return np.nan


def find_first_existing(run_dir: Path, candidates: list[str]):
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    return None


def parse_run_name(run_name: str) -> dict:
    """Extrait les métadonnées d'un run à partir de son nom de dossier."""
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

    # Résolution
    size_hw = None
    m = re.search(r"(\d{2,4})x(\d{2,4})", s)
    if m:
        size_hw = (int(m.group(1)), int(m.group(2)))

    # Epochs
    epochs = None
    m = re.search(r"(?:_e|epoch)(\d+)", s)
    if m:
        epochs = int(m.group(1))

    # Batch
    batch = None
    m = re.search(r"(?:_b|batch)(\d+)", s)
    if m:
        batch = int(m.group(1))

    # Augmentation
    aug = None
    m = re.search(r"aug(\d+)", s)
    if m:
        aug = bool(int(m.group(1)))

    aug_repeats = 1
    m = re.search(r"rep(\d+)", s)
    if m:
        aug_repeats = int(m.group(1))

    # Loss
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


def extract_metrics_from_history(hist: dict) -> dict:
    """Extrait les métriques clés depuis le history.json."""
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
def load_runs_index(exp_dir: str) -> pd.DataFrame:
    """Scan out/experiments/ et construit un DataFrame de tous les runs."""
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

        # Fichiers modèle
        best_keras = find_first_existing(rd, [
            "best.keras", "best_model.keras", f"{run_name}.keras", "model.keras",
        ])
        best_pt = find_first_existing(rd, [
            "best.pt", "best_model.pt", "model.pt",
            "best.pth", "best_model.pth", "model.pth",
        ])

        # Visuels
        pred_grid = find_first_existing(rd, [
            "pred_grid.png", "predictions.png", "overlay.png", "overlay_pred.png",
        ])
        loss_png = find_first_existing(rd, ["loss.png"])
        miou_png = find_first_existing(rd, ["miou.png", "mIoU.png", "iou.png"])

        # Métriques summary
        val_loss = safe_float(summary.get("val_loss"))
        val_mIoU = safe_float(summary.get("val_mIoU") or summary.get("val_miou"))
        test_loss = safe_float(summary.get("test_loss"))
        test_mIoU = safe_float(summary.get("test_mIoU") or summary.get("test_miou"))
        train_time_sec = safe_float(summary.get("train_time_sec"))

        # EoMT spécifique
        test_mIoU_7 = safe_float(summary.get("test_mIoU_7_no_void"))
        test_mIoU_8 = safe_float(summary.get("test_mIoU_8_including_void"))
        infer_ms_per_img = safe_float(summary.get("infer_ms_per_img"))

        # Override depuis summary.params si présent
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
            parsed["train_mode"] = parsed["train_mode"] or (
                "finetune" if params.get("trainable") else "frozen"
            )
            parsed["encoder"] = parsed["encoder"] or params.get("encoder_preset") or params.get("encoder")

        # Taille modèle
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
            "run_name": run_name,
            "run_dir": str(rd),
            **parsed,
            "val_loss": val_loss,
            "val_mIoU": val_mIoU,
            "test_loss": test_loss,
            "test_mIoU": test_mIoU,
            "train_time_sec": train_time_sec,
            "infer_ms_per_img": infer_ms_per_img,
            "test_mIoU_7_no_void": test_mIoU_7,
            "test_mIoU_8_including_void": test_mIoU_8,
            **hist_metrics,
            "best_model_path": best_model_path,
            "best_pt_path": best_pt_path,
            "has_keras": has_keras,
            "has_pytorch": has_pytorch,
            "size_mb": size_mb,
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



# HELPERS – MODÈLES / PRÉDICTION

@st.cache_resource
def load_keras_model(model_path: str):
    """Charge un modèle Keras avec les custom objects du projet."""
    if not TF_AVAILABLE:
        st.error("TensorFlow n'est pas installé.")
        return None

    custom_objects = {
        "MeanIoUArgmax": MeanIoUArgmax,
        "dice_loss_sparse": dice_loss_sparse,
    }
    # Ajoute les couches de preprocessing custom si dispo
    try:
        from scripts.models import ResNet50Preprocess
        custom_objects["ResNet50Preprocess"] = ResNet50Preprocess
    except Exception:
        pass
    try:
        from scripts.models import ConvNeXtPreprocess
        custom_objects["ConvNeXtPreprocess"] = ConvNeXtPreprocess
    except Exception:
        pass
    try:
        from scripts.models import SegFormerPreprocess
        custom_objects["SegFormerPreprocess"] = SegFormerPreprocess
    except Exception:
        pass

    try:
        model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False,
        )
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle Keras : {e}")
        return None


@st.cache_resource
def load_pytorch_model(model_path: str):
    """Charge un modèle PyTorch (.pt/.pth)."""
    if not TORCH_AVAILABLE:
        st.error("PyTorch n'est pas installé.")
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device, weights_only=False)
        if hasattr(model, "eval"):
            model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle PyTorch : {e}")
        return None


def preprocess_pil(img: Image.Image, size_hw=(256, 256)) -> np.ndarray:
    H, W = size_hw
    img = img.convert("RGB").resize((W, H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def predict_keras(img: Image.Image, model_path: str, size_hw=(256, 256), alpha=0.45):
    """Prédiction avec un modèle Keras. Retourne (img_resized, mask, mask_rgb, overlay)."""
    model = load_keras_model(model_path)
    if model is None:
        return None, None, None, None

    x = preprocess_pil(img, size_hw=size_hw)
    pred = model.predict(x[None, ...], verbose=0)[0]  # (H,W,C)
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    img_resized = Image.fromarray((x * 255).astype(np.uint8), mode="RGB")
    mask_rgb = colorize_groups(Image.fromarray(mask, mode="L"))
    over = overlay(img_resized, mask_rgb, alpha=alpha)
    return img_resized, mask, mask_rgb, over


def predict_pytorch(img: Image.Image, model_path: str, size_hw=(256, 256), alpha=0.45):
    """Prédiction avec un modèle PyTorch. Retourne (img_resized, mask, mask_rgb, overlay)."""
    model = load_pytorch_model(model_path)
    if model is None:
        return None, None, None, None

    x = preprocess_pil(img, size_hw=size_hw)
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

    with torch.no_grad():
        x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        pred = model(x_t)
        if isinstance(pred, dict):
            pred = pred.get("out", pred.get("logits", list(pred.values())[0]))
        if pred.dim() == 4:
            pred = pred[0]
        mask = pred.argmax(dim=0).cpu().numpy().astype(np.uint8)

    img_resized = Image.fromarray((x * 255).astype(np.uint8), mode="RGB")
    mask_rgb = colorize_groups(Image.fromarray(mask, mode="L"))
    over = overlay(img_resized, mask_rgb, alpha=alpha)
    return img_resized, mask, mask_rgb, over



# HELPERS – EDA

@st.cache_data
def compute_pixel_counts(df_split: pd.DataFrame, n_samples: int,
                         size_hw=(256, 256), seed=42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df_split), size=min(n_samples, len(df_split)), replace=False)
    counts = np.zeros((N_CLASSES,), dtype=np.int64)
    H, W = size_hw

    for i in idx:
        row = df_split.iloc[int(i)]
        mask_path = resolve_path(row, "mask_path", "mask_rel")
        m = remap_to_groups(load_mask_labelids(mask_path)).resize((W, H), Image.NEAREST)
        a = np.array(m, dtype=np.uint8)
        a_valid = a[a != IGNORE_LABEL]
        counts += np.bincount(a_valid.flatten(), minlength=N_CLASSES)[:N_CLASSES]

    return pd.DataFrame({"class_id": range(N_CLASSES), "class_name": CATEGORY_NAMES, "pixels": counts})


@st.cache_data
def compute_presence_counts(df_split: pd.DataFrame, n_samples: int,
                            size_hw=(256, 256), seed=42) -> pd.DataFrame:
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

    return pd.DataFrame({
        "class_id": range(N_CLASSES),
        "class_name": CATEGORY_NAMES,
        "images_with_class": present,
    })


def apply_albu_preview(img_pil, mask_pil, aug, size_hw=(256, 256), seed=0):
    """Applique une augmentation Albumentations pour prévisualisation."""
    img = img_pil.convert("RGB").resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    mask = mask_pil.resize((size_hw[1], size_hw[0]), Image.NEAREST)
    img_np = np.array(img)
    mask_np = np.array(mask, dtype=np.uint8)
    np.random.seed(seed)
    out = aug(image=img_np, mask=mask_np)
    return Image.fromarray(out["image"]), Image.fromarray(out["mask"].astype(np.uint8), mode="L")



# COMPOSANT – LÉGENDE PALETTE

def render_palette_legend():
    """Affiche une légende des 8 classes avec couleurs dans la sidebar."""
    st.sidebar.markdown("### Légende des classes")
    for i, (name, color) in enumerate(zip(CATEGORY_NAMES, PALETTE)):
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        # Contraste texte : blanc sur fond sombre, noir sur fond clair
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_c = "#ffffff" if luminance < 128 else "#000000"
        st.sidebar.markdown(
            f'<div style="background:{hex_c}; color:{text_c}; '
            f'padding:4px 10px; margin:2px 0; border-radius:4px; '
            f'font-weight:600; font-size:14px;">'
            f'{i} — {name}</div>',
            unsafe_allow_html=True,
        )



# COMPOSANT – ACCESSIBILITÉ WCAG

# Motifs de hachures Plotly (un par classe, pour ne pas dépendre de la couleur seule)
# WCAG 1.4.1 : "La couleur n'est pas le seul moyen visuel de transmettre l'info"
PATTERN_SHAPES = ["/", "\\", "x", "+", "-", "|", ".", ""]

# Palette haut contraste WCAG (pour graphiques encodeurs/modèles)
HIGH_CONTRAST_COLORS = [
    "#000000", "#D55E00", "#0072B2", "#CC79A7",
    "#009E73", "#56B4E9", "#E69F00", "#FFFFFF",
]

# Palette haut contraste par CLASSE (WCAG — mêmes classes, meilleur contraste)
HIGH_CONTRAST_CLASS_MAP = {
    "void":         "#000000",
    "flat":         "#D55E00",
    "construction": "#0072B2",
    "object":       "#CC79A7",
    "nature":       "#009E73",
    "sky":          "#56B4E9",
    "human":        "#E69F00",
    "vehicle":      "#FFFFFF",
}


def inject_wcag_css():
    """
    CSS d'accessibilité appliqué GLOBALEMENT sur toutes les pages.
    Deux niveaux :
      - Toujours actif : focus visible (2.4.7)
      - Mode WCAG toggle ON : police agrandie, espacement, contrastes renforcés
    """
    wcag_on = st.session_state.get("wcag_toggle", False)

    # ── Base : toujours actif (WCAG 2.4.7 — Focus Visible) ─────────
    base_css = """
    /* WCAG 2.4.7 — Focus visible clavier (toujours actif) */
    a:focus-visible, button:focus-visible, input:focus-visible,
    select:focus-visible, textarea:focus-visible,
    [role="button"]:focus-visible, [tabindex]:focus-visible {
        outline: 3px solid #4da6ff !important;
        outline-offset: 2px !important;
    }
    """

    # ── Mode WCAG ON :───────
    wcag_css = ""
    if wcag_on:
        wcag_css = """
    /* ═══════════════════════════════════════════════════════════
       MODE ACCESSIBLE ACTIVÉ — Changements visibles globaux
       ═══════════════════════════════════════════════════════════ */

    /* WCAG 1.4.4 — Texte redimensionnable (base 18px au lieu de 14px) */
    html, body, [class*="css"],
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stText, .stCaption, .stDataFrame,
    [data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }

    /* WCAG 1.4.12 — Espacement du texte (letter-spacing, word-spacing) */
    p, li, td, th, label, span, div {
        letter-spacing: 0.02em !important;
        word-spacing: 0.08em !important;
    }

    /* Titres plus gros et plus espacés */
    h1 { font-size: 2.4rem !important; }
    h2 { font-size: 2.0rem !important; }
    h3 { font-size: 1.6rem !important; }
    h1, h2, h3 { margin-bottom: 0.8em !important; }

    /* Labels et widgets — police renforcée */
    label, .stSelectbox label, .stSlider label,
    .stRadio label, .stCheckbox label {
        font-weight: 700 !important;
        font-size: 18px !important;
    }

    /* Métriques grandes et lisibles */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Boutons plus grands et plus visibles */
    .stButton > button {
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 0.7rem 1.4rem !important;
        border-width: 2px !important;
    }

    /* Focus encore plus visible en mode WCAG */
    a:focus-visible, button:focus-visible, input:focus-visible,
    select:focus-visible, textarea:focus-visible,
    [role="button"]:focus-visible, [tabindex]:focus-visible {
        outline: 4px solid #ff6600 !important;
        outline-offset: 3px !important;
        box-shadow: 0 0 0 6px rgba(255, 102, 0, 0.3) !important;
    }

    /* Selectbox et inputs plus grands */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        font-size: 18px !important;
        min-height: 48px !important;
    }

    /* Captions plus lisibles (pas trop petites) */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 16px !important;
    }

    /* Tableaux — texte plus grand, lignes plus espacées */
    .stDataFrame td, .stDataFrame th {
        font-size: 16px !important;
        padding: 8px 12px !important;
    }

    /* Sidebar — texte plus grand aussi */
    [data-testid="stSidebar"] {
        font-size: 17px !important;
    }
    [data-testid="stSidebar"] label {
        font-size: 17px !important;
    }
        """

    st.markdown(f"<style>{base_css}{wcag_css}</style>", unsafe_allow_html=True)

    # Bandeau visuel quand le mode est actif
    if wcag_on:
        st.markdown(
            '<div style="background:#005fcc; color:white; padding:6px 16px; '
            'border-radius:4px; font-weight:600; font-size:14px; '
            'margin-bottom:10px; text-align:center;">'
            '♿ Mode WCAG activé'
            '</div>',
            unsafe_allow_html=True,
        )


def _get_wcag_mode() -> bool:
    """Vérifie si le mode WCAG est activé (set dans la sidebar)."""
    return st.session_state.get("wcag_toggle", False)


def make_accessible_bar(df, x, y, title, text_col=None, color_col=None):
    wcag = _get_wcag_mode()
    is_class = (color_col == "class_name")

    # Choisi la bonne palette
    if is_class:
        cmap = HIGH_CONTRAST_CLASS_MAP if wcag else CLASS_COLOR_MAP
        fig = px.bar(
            df, x=x, y=y, text=text_col or y,
            title=title, color=color_col,
            color_discrete_map=cmap,
        )
    else:
        colors = HIGH_CONTRAST_COLORS if wcag else PLOTLY_COLORS
        fig = px.bar(
            df, x=x, y=y, text=text_col or y,
            title=title, color=color_col,
            color_discrete_sequence=colors,
        )

    fig.update_traces(textposition="outside", textfont_size=12)

    if wcag:
        for i, trace in enumerate(fig.data):
            trace.marker.pattern.shape = PATTERN_SHAPES[i % len(PATTERN_SHAPES)]
            trace.marker.pattern.solidity = 0.6
            trace.marker.line.width = 1.5
            trace.marker.line.color = "white"

    fig.update_layout(
        xaxis_title=x, yaxis_title=y,
        xaxis_tickangle=-35,
        font=dict(size=13),
        title_font_size=16,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=14),
    )
    return fig


def make_accessible_scatter(df, x, y, title, color_col=None, symbol_col=None,
                            hover_data=None):
    """
    Scatter Plotly accessible :
    - WCAG 1.4.1 : forme (symbol) différente par catégorie
    - WCAG 1.4.11 : bordure sur les points (mode WCAG)
    """
    wcag = _get_wcag_mode()
    colors = HIGH_CONTRAST_COLORS if wcag else PLOTLY_COLORS

    fig = px.scatter(
        df, x=x, y=y,
        color=color_col,
        symbol=symbol_col,
        hover_data=hover_data,
        title=title,
        color_discrete_sequence=colors,
    )

    if wcag:
        fig.update_traces(
            marker=dict(size=12, line=dict(width=2, color="white")),
        )
    else:
        fig.update_traces(marker=dict(size=9))

    fig.update_layout(
        font=dict(size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=14),
    )
    return fig


def render_chart_alt_text(description: str):
    """
    WCAG 1.1.1 — Alternative textuelle pour un graphique.
    Affiche une description masquable sous le graphique.
    """
    with st.expander("Description du graphique (accessibilité)", expanded=False):
        st.write(description)



# PAGE 1 — EDA

def render_eda(train_df, val_df, test_df, size_hw, alpha):
    st.title("EDA — Cityscapes (8 classes)")

    # Métriques principales
    c1, c2, c3 = st.columns(3)
    c1.metric("Train", len(train_df))
    c2.metric("Val", len(val_df))
    c3.metric("Test", len(test_df))

    # Exemples d'images
    st.subheader("Exemples d'images + masques (remap 8 classes)")
    split_choice = st.selectbox(
        "Split", ["train", "val", "test"], index=2, key="eda_split",
        help="Sélectionnez le split à explorer.",
    )
    n_show = st.slider("Nombre d'exemples", 2, 18, 6, 1, key="eda_nshow")

    df_split = {"train": train_df, "val": val_df, "test": test_df}[split_choice]
    df_split = df_split.reset_index(drop=True)

    cols = st.columns(3)
    for i in range(min(n_show, len(df_split))):
        row = df_split.iloc[i]
        img_path = resolve_path(row, "image_path", "image_rel")
        mask_path = resolve_path(row, "mask_path", "mask_rel")

        img = load_rgb(img_path).resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        m = remap_to_groups(load_mask_labelids(mask_path)).resize((size_hw[1], size_hw[0]), Image.NEAREST)
        m_rgb = colorize_groups(m)
        ov = overlay(img, m_rgb, alpha=alpha)

        with cols[i % 3]:
            st.image(img, caption=f"Image #{i} ({split_choice})", use_container_width=True)
            st.image(m_rgb, caption=f"Masque 8 classes #{i}", use_container_width=True)
            st.image(ov, caption=f"Overlay #{i}", use_container_width=True)

    # Comptages interactifs
    st.subheader("Comptages interactifs — pixels par classe")
    n_samples = st.slider(
        "Échantillon (nb masques)", 50, 1500, 200, 50, key="eda_nsamples",
        help="Nombre de masques utilisés pour le comptage (échantillonnage aléatoire).",
    )
    seed = st.number_input("Seed", value=42, step=1, key="eda_seed")

    counts_df = compute_pixel_counts(df_split, n_samples=n_samples, size_hw=size_hw, seed=int(seed))
    fig_px = make_accessible_bar(
        counts_df, x="class_name", y="pixels",
        title=f"Pixels par classe (split={split_choice}, n={min(n_samples, len(df_split))})",
        color_col="class_name",
    )
    st.plotly_chart(fig_px, use_container_width=True)
    # WCAG 1.1.1 — Alternative textuelle
    top_class = counts_df.loc[counts_df["pixels"].idxmax(), "class_name"]
    render_chart_alt_text(
        f"Diagramme en barres du nombre de pixels par classe sur {min(n_samples, len(df_split))} masques "
        f"du split {split_choice}. La classe la plus représentée est '{top_class}'."
    )
    st.dataframe(counts_df, use_container_width=True, hide_index=True)

    # Présence des classes
    st.subheader("Présence des classes — images contenant la classe")
    presence_df = compute_presence_counts(df_split, n_samples=n_samples, size_hw=size_hw, seed=int(seed))
    fig_pr = make_accessible_bar(
        presence_df, x="class_name", y="images_with_class",
        title="Images contenant chaque classe (échantillon)",
        color_col="class_name",
    )
    st.plotly_chart(fig_pr, use_container_width=True)
    # WCAG 1.1.1 — Alternative textuelle
    top_presence = presence_df.loc[presence_df["images_with_class"].idxmax(), "class_name"]
    render_chart_alt_text(
        f"Diagramme en barres du nombre d'images contenant chaque classe. "
        f"La classe présente dans le plus d'images est '{top_presence}'."
    )
    st.dataframe(presence_df, use_container_width=True, hide_index=True)

    # Transformations
    st.subheader("Transformations — exemples (equalisation, floutage)")
    idx_t = st.slider("Index exemple", 0, max(0, len(df_split) - 1), 0, 1, key="eda_idx_transform")
    row = df_split.iloc[int(idx_t)]
    img_path = resolve_path(row, "image_path", "image_rel")
    img = load_rgb(img_path).resize((size_hw[1], size_hw[0]), Image.BILINEAR)

    t1, t2, t3 = st.columns(3)
    with t1:
        st.image(img, caption="Original", use_container_width=True)
    with t2:
        st.image(ImageOps.equalize(img), caption="Equalisation d'histogramme", use_container_width=True)
    with t3:
        st.image(
            img.filter(ImageFilter.GaussianBlur(radius=2)),
            caption="Flou gaussien (r=2)", use_container_width=True,
        )

    # Aperçu augmentations
    st.subheader("Aperçu des augmentations (Albumentations)")
    show_aug = st.checkbox("Afficher les augmentations", value=True, key="eda_show_aug")
    if show_aug:
        idx_a = st.slider("Index exemple (split)", 0, max(0, len(df_split) - 1), 0, 1, key="eda_idx_aug")
        n_aug = st.slider("Nombre de variantes", 1, 9, 4, 1, key="eda_n_aug")

        row = df_split.iloc[int(idx_a)]
        img_path = resolve_path(row, "image_path", "image_rel")
        mask_path = resolve_path(row, "mask_path", "mask_rel")
        img0 = load_rgb(img_path)
        m0 = remap_to_groups(load_mask_labelids(mask_path))
        aug = make_train_aug()

        grid_cols = st.columns(min(3, n_aug))
        for j in range(n_aug):
            img_aug, m_aug = apply_albu_preview(img0, m0, aug, size_hw=size_hw, seed=100 + j)
            m_aug_rgb = colorize_groups(m_aug)
            ov = overlay(img_aug, m_aug_rgb, alpha=alpha)
            with grid_cols[j % min(3, n_aug)]:
                st.image(img_aug, caption=f"Augmentation #{j}", use_container_width=True)
                st.image(ov, caption=f"Overlay aug #{j}", use_container_width=True)



# HELPER – IoU PAR CLASSE

def compute_iou_per_class(gt_mask: np.ndarray, pred_mask: np.ndarray) -> pd.DataFrame:
    """Calcule l'IoU par classe entre GT et prédiction (arrays 2D, 0..7)."""
    rows = []
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()
    # Exclure les pixels IGNORE_LABEL
    valid = gt_flat != IGNORE_LABEL
    gt_v = gt_flat[valid]
    pred_v = pred_flat[valid]

    ious = []
    for c in range(N_CLASSES):
        gt_c = (gt_v == c)
        pred_c = (pred_v == c)
        inter = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        iou = float(inter / union) if union > 0 else 0.0
        ious.append(iou)
        rows.append({"class_id": c, "class_name": CATEGORY_NAMES[c], "IoU": round(iou, 4)})

    miou = float(np.mean(ious))
    return pd.DataFrame(rows), miou


def render_prediction_stats(pred_mask, gt_mask=None):
    """Affiche les stats de prédiction : distribution + IoU si GT disponible."""
    wcag = _get_wcag_mode()
    cmap = HIGH_CONTRAST_CLASS_MAP if wcag else CLASS_COLOR_MAP

    # Distribution des classes prédites
    unique, counts = np.unique(pred_mask, return_counts=True)
    stats_df = pd.DataFrame({
        "class_id": unique,
        "class_name": [CATEGORY_NAMES[int(c)] for c in unique],
        "pixels": counts,
        "pct": (counts / counts.sum() * 100).round(2),
    })

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Distribution des classes (prédiction)**")
        fig_dist = px.bar(
            stats_df, x="class_name", y="pct", text="pct",
            title="% pixels par classe",
            color="class_name", color_discrete_map=cmap,
        )
        fig_dist.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        if wcag:
            for i, trace in enumerate(fig_dist.data):
                trace.marker.pattern.shape = PATTERN_SHAPES[i % len(PATTERN_SHAPES)]
                trace.marker.pattern.solidity = 0.6
                trace.marker.line.width = 1.5
                trace.marker.line.color = "white"
        fig_dist.update_layout(
            showlegend=False, yaxis_title="% pixels",
            xaxis_title="", xaxis_tickangle=-25,
            font=dict(size=12),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        if gt_mask is not None:
            iou_df, miou = compute_iou_per_class(gt_mask, pred_mask)
            st.markdown(f"**IoU par classe** — mIoU = **{miou:.4f}**")
            fig_iou = px.bar(
                iou_df, x="class_name", y="IoU", text="IoU",
                title=f"IoU par classe (mIoU = {miou:.4f})",
                color="class_name", color_discrete_map=cmap,
            )
            fig_iou.update_traces(textposition="outside", texttemplate="%{text:.3f}")
            if wcag:
                for i, trace in enumerate(fig_iou.data):
                    trace.marker.pattern.shape = PATTERN_SHAPES[i % len(PATTERN_SHAPES)]
                    trace.marker.pattern.solidity = 0.6
                    trace.marker.line.width = 1.5
                    trace.marker.line.color = "white"
            fig_iou.update_layout(
                showlegend=False, yaxis_title="IoU", yaxis_range=[0, 1],
                xaxis_title="", xaxis_tickangle=-25,
                font=dict(size=12),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig_iou, use_container_width=True)
            st.dataframe(iou_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("**Distribution détaillée**")
            st.dataframe(stats_df, use_container_width=True, hide_index=True)



# PAGE 2 — PRÉDICTION

def render_prediction(exp_df, test_df, size_hw, alpha):
    st.title("Prédiction — image + modèle")

    # Filtrer les runs qui ont un modèle chargeable
    exp_pred = exp_df[exp_df["has_keras"] | exp_df["has_pytorch"]].copy()
    if len(exp_pred) == 0:
        st.warning(
            "Aucun modèle (.keras ou .pt) trouvé dans out/experiments/. "
            "Vérifiez que vos runs contiennent un fichier best.keras ou best.pt."
        )
        # Affiche quand même les pred_grid disponibles
        exp_with_grid = exp_df[exp_df["pred_grid_png"].astype(str).str.len() > 0]
        if len(exp_with_grid) > 0:
            st.subheader("Résultats pré-calculés (pred_grid)")
            run_choice = st.selectbox("Run", exp_with_grid["run_name"].tolist(), key="pred_grid_run")
            row = exp_with_grid[exp_with_grid["run_name"] == run_choice].iloc[0]
            p = row["pred_grid_png"]
            if p and Path(p).exists():
                st.image(Image.open(p), caption=f"Prédictions — {run_choice}", use_container_width=True)
        return

    run_choice = st.selectbox(
        "Modèle (run)", exp_pred["run_name"].tolist(), index=0, key="pred_run",
        help="Sélectionnez un run d'entraînement pour faire la prédiction.",
    )
    run_row = exp_pred[exp_pred["run_name"] == run_choice].iloc[0]

    # Déterminer le type de modèle
    is_keras = run_row["has_keras"]
    is_pytorch = run_row["has_pytorch"]
    model_path = run_row["best_model_path"] if is_keras else run_row["best_pt_path"]

    backend_str = "Keras" if is_keras else "PyTorch"
    st.caption(f"Backend: **{backend_str}** — Modèle: `{model_path}`")

    # Métriques du run sélectionné
    m1, m2, m3 = st.columns(3)
    m1.metric("test mIoU", f"{run_row.get('test_mIoU', np.nan):.4f}" if pd.notna(run_row.get("test_mIoU")) else "N/A")
    m2.metric("val mIoU", f"{run_row.get('val_mIoU', np.nan):.4f}" if pd.notna(run_row.get("val_mIoU")) else "N/A")
    m3.metric("Temps train", f"{run_row.get('train_time_sec', np.nan)/60:.1f} min" if pd.notna(run_row.get("train_time_sec")) else "N/A")

    mode = st.radio(
        "Source image", ["Test (dropdown)", "Upload"],
        horizontal=True, key="pred_source",
        help="Choisissez une image du set de test ou uploadez la vôtre.",
    )

    predict_fn = predict_keras if is_keras else predict_pytorch

    if mode == "Test (dropdown)":
        test_df2 = test_df.reset_index(drop=True)
        idx = st.slider("Index image test", 0, max(0, len(test_df2) - 1), 0, 1, key="pred_idx")
        row = test_df2.iloc[int(idx)]
        img_path = resolve_path(row, "image_path", "image_rel")
        mask_path = resolve_path(row, "mask_path", "mask_rel")

        img = load_rgb(img_path)
        gt = remap_to_groups(load_mask_labelids(mask_path)).resize((size_hw[1], size_hw[0]), Image.NEAREST)
        gt_rgb = colorize_groups(gt)

        st.image(
            img.resize((size_hw[1], size_hw[0])),
            caption=f"Image test #{idx}", use_container_width=True,
        )

        if st.button("Prédire", key="pred_btn"):
            with st.spinner("Inférence en cours..."):
                result = predict_fn(img, model_path, size_hw=size_hw, alpha=alpha)

            if result[0] is not None:
                img_resized, pred_mask, pred_rgb, pred_ov = result
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(img_resized, caption="Entrée (resize)", use_container_width=True)
                with c2:
                    st.image(pred_rgb, caption="Prédiction", use_container_width=True)
                with c3:
                    st.image(pred_ov, caption="Overlay prédiction", use_container_width=True)

                c4, c5 = st.columns(2)
                with c4:
                    st.image(gt_rgb, caption="Vérité terrain (GT)", use_container_width=True)
                with c5:
                    st.image(overlay(img_resized, gt_rgb, alpha=alpha), caption="Overlay GT", use_container_width=True)

                # Stats + IoU par classe (avec le GT ici)
                gt_np = np.array(gt, dtype=np.uint8)
                render_prediction_stats(pred_mask, gt_mask=gt_np)
    else:
        up = st.file_uploader(
            "Uploadez une image", type=["jpg", "jpeg", "png"], key="pred_upload",
            help="Format JPG ou PNG, toute résolution.",
        )
        if up is not None:
            img = Image.open(up).convert("RGB")
            st.image(img, caption="Image uploadée", use_container_width=True)
            if st.button("Prédire", key="pred_btn_up"):
                with st.spinner("Inférence en cours..."):
                    result = predict_fn(img, model_path, size_hw=size_hw, alpha=alpha)
                if result[0] is not None:
                    img_resized, pred_mask, pred_rgb, pred_ov = result
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(img_resized, caption="Entrée (resize)", use_container_width=True)
                    with c2:
                        st.image(pred_rgb, caption="Prédiction", use_container_width=True)
                    with c3:
                        st.image(pred_ov, caption="Overlay", use_container_width=True)

                    # Stats sans GT (pas de IoU possible)
                    render_prediction_stats(pred_mask, gt_mask=None)



# PAGE 3 — COMPARAISON

def render_comparison(exp_df: pd.DataFrame):
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
        "size_hw", "epochs", "batch", "aug", "aug_repeats", "loss_name",
        "score_main", "test_mIoU", "best_val_mIoU", "val_mIoU",
        "best_val_mIoU_epoch", "best_val_loss", "best_val_loss_epoch",
        "train_time_sec", "infer_ms_per_img",
        "val_mIoU_tail_mean", "val_mIoU_tail_std",
        "last_lr", "n_epochs_ran", "size_mb",
    ]
    available = [c for c in default_cols if c in df.columns]
    cols = st.multiselect("Colonnes affichées", df.columns.tolist(), default=available, key="cmp_cols")

    view = df[cols].copy() if cols else df.copy()
    st.dataframe(view.fillna("—"), use_container_width=True, height=520)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger CSV", data=csv, file_name="comparison_runs.csv", mime="text/csv",
        help="Export du tableau filtré au format CSV.",
    )

    # Graphiques
    st.subheader("Graphiques")
    c4, c5 = st.columns(2)

    with c4:
        metric = st.selectbox(
            "Métrique (bar chart)",
            ["score_main", "best_val_mIoU", "best_val_loss", "train_time_sec"],
            index=0, key="cmp_metric",
        )
        N = st.slider("Top N", 5, 30, 12, 1, key="cmp_topn")
        dd = df.copy()
        if metric in dd.columns:
            dd = dd[np.isfinite(dd[metric].astype(float))]
            asc = metric in ("best_val_loss", "train_time_sec")
            dd = dd.sort_values(metric, ascending=asc).head(int(N))

        if len(dd) > 0:
            fig = make_accessible_bar(
                dd, x="run_name", y=metric,
                title=f"Top {N} — {metric}",
                color_col="encoder",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de données suffisantes pour ce graphique.")

    with c5:
        scatter_x = st.selectbox("Axe X", ["train_time_sec", "batch", "epochs", "size_mb"], index=0, key="cmp_x")
        scatter_y = st.selectbox("Axe Y", ["score_main", "best_val_mIoU", "val_mIoU"], index=0, key="cmp_y")
        dd = df.copy()
        if scatter_x in dd.columns and scatter_y in dd.columns:
            dd = dd[np.isfinite(dd[scatter_x].astype(float)) & np.isfinite(dd[scatter_y].astype(float))]

        if len(dd) > 0:
            fig2 = make_accessible_scatter(
                dd, x=scatter_x, y=scatter_y,
                title=f"{scatter_y} vs {scatter_x}",
                color_col="encoder",
                symbol_col="model_family",
                hover_data=["run_name", "train_mode", "loss_name"],
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pas de points suffisants pour le scatter.")

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

        fig_box = px.box(
            dd, x=group_col, y=metric2,
            color=group_col,
            title=f"Distribution {metric2} par {group_col}",
            color_discrete_sequence=box_colors,
            points="all",
        )
        if wcag:
            fig_box.update_traces(marker=dict(line=dict(width=1.5, color="white")))
        fig_box.update_layout(
            font=dict(size=13),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Données insuffisantes pour le résumé par groupe.")

    # Courbes d'entraînement (selon la run sélectionné)
    st.subheader("Détails d'un run")
    run_choice = st.selectbox("Run", df["run_name"].tolist(), index=0, key="cmp_run")
    row = df[df["run_name"] == run_choice].iloc[0]
    run_dir = Path(row["run_dir"])

    # Charge le history.json pour tracer les courbes
    history = _safe_read_json(run_dir / "history.json")
    if history:
        tab_loss, tab_miou = st.tabs(["Courbe Loss", "Courbe mIoU"])

        with tab_loss:
            if "loss" in history and "val_loss" in history:
                epochs_range = list(range(1, len(history["loss"]) + 1))
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=epochs_range, y=history["loss"],
                    mode="lines", name="Train loss",
                    line=dict(color="#2196F3", width=2),
                ))
                fig_loss.add_trace(go.Scatter(
                    x=epochs_range, y=history["val_loss"],
                    mode="lines", name="Val loss",
                    line=dict(color="#F44336", width=2),
                ))
                fig_loss.update_layout(
                    title=f"Loss — {run_choice}",
                    xaxis_title="Epoch", yaxis_title="Loss",
                    font=dict(size=13),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.info("Pas de données de loss pour ce run.")

        with tab_miou:
            miou_key = "mIoU" if "mIoU" in history else "miou" if "miou" in history else None
            vmiou_key = "val_mIoU" if "val_mIoU" in history else "val_miou" if "val_miou" in history else None

            if miou_key and vmiou_key and miou_key in history and vmiou_key in history:
                epochs_range = list(range(1, len(history[miou_key]) + 1))
                fig_miou = go.Figure()
                fig_miou.add_trace(go.Scatter(
                    x=epochs_range, y=history[miou_key],
                    mode="lines", name="Train mIoU",
                    line=dict(color="#4CAF50", width=2),
                ))
                fig_miou.add_trace(go.Scatter(
                    x=epochs_range, y=history[vmiou_key],
                    mode="lines", name="Val mIoU",
                    line=dict(color="#FF9800", width=2),
                ))
                fig_miou.update_layout(
                    title=f"mIoU — {run_choice}",
                    xaxis_title="Epoch", yaxis_title="mIoU",
                    font=dict(size=13),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_miou, use_container_width=True)
            else:
                st.info("Pas de données mIoU pour ce run.")

    # Visuels sauvegardés
    cols_img = st.columns(3)
    for i, (key, label) in enumerate([
        ("pred_grid_png", "Grille de prédictions"),
        ("loss_png", "Courbe Loss (image)"),
        ("miou_png", "Courbe mIoU (image)"),
    ]):
        p = row.get(key, "")
        with cols_img[i]:
            if p and Path(p).exists():
                st.image(Image.open(p), caption=label, use_container_width=True)
            else:
                st.caption(f"{label} — non disponible")


# PAGE 4 — À PROPOS


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

    Le mode **WCAG** (activable dans la sidebar) applique les critères suivants sur l'ensemble du dashboard :

    | Critère WCAG | Description | Implémentation |
    |-------------|-------------|----------------|
    | **1.1.1** Non-text Content | Alternatives textuelles | Description textuelle sous chaque graphique + tableau de données associé |
    | **1.4.1** Use of Color | La couleur n'est pas le seul moyen | Hachures/motifs distincts sur les barres + formes différentes sur les scatter plots |
    | **1.4.3** Contrast (Minimum) | Ratio de contraste ≥ 4.5:1 | Palette haut contraste + valeurs numériques sur chaque élément |
    | **1.4.4** Resize Text | Texte redimensionnable à 200% | Police agrandie à 18px, titres proportionnellement plus grands |
    | **1.4.11** Non-text Contrast | Contraste des éléments graphiques | Bordures visibles sur barres et points |
    | **1.4.12** Text Spacing | Espacement du texte | Letter-spacing et word-spacing augmentés |
    | **2.4.7** Focus Visible | Indicateur de focus clavier | Outline orange de 4px + halo sur tous les éléments interactifs |

    *Le focus clavier (2.4.7) est actif en permanence, même sans le mode WCAG.*
    """)

    # Légende palette
    st.subheader("Palette des 8 classes")

    palette_data = []
    for i, (name, color) in enumerate(zip(CATEGORY_NAMES, PALETTE)):
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        palette_data.append({
            "ID": i, "Classe": name, "Couleur (hex)": hex_c,
            "R": color[0], "G": color[1], "B": color[2],
        })
    st.dataframe(pd.DataFrame(palette_data), use_container_width=True, hide_index=True)

    cols = st.columns(8)
    for i, (name, color) in enumerate(zip(CATEGORY_NAMES, PALETTE)):
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_c = "#fff" if luminance < 128 else "#000"
        with cols[i]:
            st.markdown(
                f'<div style="background:{hex_c}; color:{text_c}; '
                f'text-align:center; padding:20px 5px; border-radius:8px; '
                f'font-weight:700; font-size:13px; min-height:80px; '
                f'display:flex; align-items:center; justify-content:center;">'
                f'{name}</div>',
                unsafe_allow_html=True,
            )


# MAIN — SIDEBAR + ROUTING

# Charge les données
df_all = load_split_df()
train_df, val_df, test_df = get_split_dfs(df_all)
exp_df = load_runs_index(str(EXP_DIR))

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Page", ["EDA", "Prédiction", "Comparaison", "À propos"],
    index=0, key="nav_page",
)

st.sidebar.markdown("---")

# Résolution (256 ou 512)
size_options = [(256, 256), (512, 512)]
size_labels = ["256×256", "512×512"]
size_idx = st.sidebar.selectbox(
    "Résolution (H×W)", range(len(size_options)),
    format_func=lambda i: size_labels[i],
    index=0, key="size_select",
)
size_hw = size_options[size_idx]

alpha = st.sidebar.slider("Alpha overlay", 0.0, 1.0, 0.45, 0.05, key="alpha_slider")

# Accessibilité — CSS focus visible toujours actif (WCAG 2.4.7)
inject_wcag_css()

# Toggle pour le mode accessibilité
accessible_mode = st.sidebar.checkbox(
    "♿ Mode accessible (WCAG)",
    value=False, key="wcag_toggle",
    help="Active l'accessibilité renforcée : police agrandie (18px), "
         "espacement du texte, hachures sur les graphiques, palette haut contraste, "
         "bordures visibles, focus orange renforcé. S'applique à toutes les pages.",
)

st.sidebar.markdown("---")

# Légende palette (toujours visible dans sidebar)
render_palette_legend()

# Infos frameworks
st.sidebar.markdown("---")
st.sidebar.caption(
    f"TensorFlow: {'✅' if TF_AVAILABLE else '❌'}  |  "
    f"PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}"
)
st.sidebar.caption(f"Runs détectés: {len(exp_df)}")

# Routing
if page == "EDA":
    render_eda(train_df, val_df, test_df, size_hw, alpha)
elif page == "Prédiction":
    render_prediction(exp_df, test_df, size_hw, alpha)
elif page == "Comparaison":
    render_comparison(exp_df)
else:
    render_about()