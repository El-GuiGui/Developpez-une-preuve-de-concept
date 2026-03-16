from __future__ import annotations

import os
from pathlib import Path

SPLIT_CSV_NAME = "cityscapes_split_testXtrainXval.csv"


def _find_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path.cwd()

    env_root = os.getenv("PROJ8_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "scripts").exists() and (p / "data").exists():
            return p

    start = start.resolve()

    for p in [start] + list(start.parents):
        if (p / "scripts").exists() and (p / "data").exists():
            return p

    if start.name == "notebooks" and (start.parent / "scripts").exists():
        return start.parent

    raise RuntimeError(
        "Impossible de trouver la racine PROJ8. "
        "Définis PROJ8_ROOT=/chemin/vers/PROJ8 ou lance le notebook depuis PROJ8/notebooks."
    )


PROJECT_ROOT = _find_project_root()

# Dossiers standards
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CITYSCAPES_DIR = RAW_DIR / "cityscapes"

OUT_DIR = PROJECT_ROOT / "out"
EXP_DIR = OUT_DIR / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
LOGS_DIR = PROJECT_ROOT / "logs"

# Cityscapes
GT_FINE_DIR = CITYSCAPES_DIR / "gtFine"
LEFTIMG_DIR = CITYSCAPES_DIR / "leftImg8bit"

# Split CSV
CSV_SPLIT = OUT_DIR / SPLIT_CSV_NAME

_SPLIT_FALLBACKS = [
    OUT_DIR / "cityscapes_split_test_x_train_x_val.csv",
    OUT_DIR / "cityscapes_split_testXtrainXval.csv",
    OUT_DIR / "cityscapes_split_testxtrainxval.csv",
]


def resolve_split_csv() -> Path:
    if CSV_SPLIT.exists():
        return CSV_SPLIT
    for p in _SPLIT_FALLBACKS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Split CSV introuvable. Attendu: {CSV_SPLIT} "
        f"(ou fallbacks: {[str(x) for x in _SPLIT_FALLBACKS]})."
    )


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
