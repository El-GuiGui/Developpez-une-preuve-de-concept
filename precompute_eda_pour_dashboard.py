"""
Pré-calcule toutes les stats EDA sur le dataset Cityscapes
et extrait un échantillon d'images pour le dashboard déployé.

Usage (via la racine du proj):
    python precompute_eda.py

Produit :
    app_dashboard/eda_cache/stats.json       — stats calculées sur TOUT le dataset
    app_dashboard/eda_cache/samples/          — ~30 images+masques redimensionnés
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import resolve_split_csv, CITYSCAPES_DIR
from scripts.preprocessing import (
    load_rgb, load_mask_labelids, remap_to_groups, colorize_groups, overlay,
    CATEGORY_NAMES, IGNORE_LABEL, N_CLASSES,
)

# Config
OUT_DIR = ROOT / "app_dashboard" / "eda_cache"
SAMPLES_DIR = OUT_DIR / "samples"
SIZE_HW = (256, 256)
N_SAMPLES_PER_SPLIT = 10
SEED = 42


def resolve_path(row, col_abs, col_rel):
    if col_abs in row and isinstance(row[col_abs], str) and len(row[col_abs]) > 0:
        return row[col_abs]
    return f"{CITYSCAPES_DIR}/{row[col_rel]}"


def compute_stats_full(df_split: pd.DataFrame, split_name: str):
    """Calcule les stats sur les images du split."""
    H, W = SIZE_HW
    pixel_counts = np.zeros((N_CLASSES,), dtype=np.int64)
    presence_counts = np.zeros((N_CLASSES,), dtype=np.int64)
    total = len(df_split)

    for i in range(total):
        if i % 100 == 0:
            print(f"  [{split_name}] {i}/{total}...")
        row = df_split.iloc[i]
        mask_path = resolve_path(row, "mask_path", "mask_rel")
        m = remap_to_groups(load_mask_labelids(mask_path)).resize((W, H), Image.NEAREST)
        a = np.array(m, dtype=np.uint8)
        valid = a[a != IGNORE_LABEL]

        pixel_counts += np.bincount(valid.flatten(), minlength=N_CLASSES)[:N_CLASSES]

        for c in np.unique(valid):
            if 0 <= int(c) < N_CLASSES:
                presence_counts[int(c)] += 1

    return {
        "pixel_counts": pixel_counts.tolist(),
        "presence_counts": presence_counts.tolist(),
        "n_images": total,
    }


def extract_samples(df_split: pd.DataFrame, split_name: str, n: int):
    """Extrait n images + masques redimensionnés."""
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(df_split), size=min(n, len(df_split)), replace=False)
    indices.sort()
    H, W = SIZE_HW

    saved = []
    for j, idx in enumerate(indices):
        row = df_split.iloc[int(idx)]
        img_path = resolve_path(row, "image_path", "image_rel")
        mask_path = resolve_path(row, "mask_path", "mask_rel")

        img = load_rgb(img_path).resize((W, H), Image.BILINEAR)
        mask = remap_to_groups(load_mask_labelids(mask_path)).resize((W, H), Image.NEAREST)

        img_name = f"{split_name}_{j:03d}_img.png"
        mask_name = f"{split_name}_{j:03d}_mask.png"

        img.save(SAMPLES_DIR / img_name)
        mask.save(SAMPLES_DIR / mask_name)

        saved.append({"img": img_name, "mask": mask_name, "split": split_name})

    return saved


def main():
    print("Chargement du CSV...")
    csv_path = resolve_split_csv()
    df = pd.read_csv(csv_path)

    train_df = df[df["split_final"] == "train"].copy()
    val_df = df[df["split_final"] == "val"].copy()
    test_df = df[df["split_final"] == "test"].copy()

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Créer les dossiers
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Stats complètes
    print("\nCalcul des stats (dataset complet)...")
    stats = {
        "class_names": CATEGORY_NAMES,
        "n_classes": N_CLASSES,
        "size_hw": list(SIZE_HW),
    }

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n--- Split: {name} ({len(split_df)} images) ---")
        stats[name] = compute_stats_full(split_df, name)

    # Extraction des samples
    print("\nExtraction des échantillons visuels...")
    all_samples = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        samples = extract_samples(split_df, name, N_SAMPLES_PER_SPLIT)
        all_samples.extend(samples)
        print(f"  {name}: {len(samples)} images sauvées")

    stats["samples"] = all_samples

    # Sauvegarde
    stats_path = OUT_DIR / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStats sauvées : {stats_path}")
    print(f"Samples sauvés : {SAMPLES_DIR} ({len(all_samples)} fichiers)")

    # Taille totale
    total_bytes = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print(f"Taille totale eda_cache : {total_bytes / 1024 / 1024:.1f} Mo")


if __name__ == "__main__":
    main()
