import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image


import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_run_name(run_name: str) -> dict:
    import re
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


def extract_metrics_from_history(hist: dict) -> dict:
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


def safe_float(x):
    try:
        return float(x) if x is not None else np.nan
    except (ValueError, TypeError):
        return np.nan

class TestParseRunName:

    def test_unet_scratch(self):
        r = parse_run_name("UNET_SCRATCH_256x256_b4_aug1_rep1_ce_dice_e60_seed42")
        assert r["model_family"] == "unet"
        assert r["encoder"] == "scratch"
        assert r["train_mode"] == "scratch"
        assert r["size_hw"] == (256, 256)
        assert r["batch"] == 4
        assert r["epochs"] == 60
        assert r["aug"] is True
        assert r["aug_repeats"] == 1
        assert r["loss_name"] == "ce_dice"

    def test_unet_vgg16_frozen(self):
        r = parse_run_name("UNET_VGG16_256x256_b4_aug1_rep1_ce_dice_frozen_e40_seed42")
        assert r["model_family"] == "unet"
        assert r["encoder"] == "vgg16"
        assert r["train_mode"] == "frozen"
        assert r["size_hw"] == (256, 256)
        assert r["epochs"] == 40
        assert r["loss_name"] == "ce_dice"

    def test_unet_resnet50_finetune(self):
        r = parse_run_name("UNET_RESNET50_256x256_b4_aug1_rep1_ce_dice_finetune_e40_seed42")
        assert r["model_family"] == "unet"
        assert r["encoder"] == "resnet50"
        assert r["train_mode"] == "finetune"

    def test_convnext(self):
        r = parse_run_name("UNET_CONVNEXT_TINY_256x256_b4_aug1_rep1_ce_dice_frozen_e40_seed42")
        assert r["model_family"] == "unet"
        assert r["encoder"] == "convnext_tiny"
        assert r["train_mode"] == "frozen"

    def test_segformer(self):
        r = parse_run_name("SEGFORMER_MIT-B0_256x256_b4_aug1_rep1_ce_dice_frozen_e40_seed42")
        assert r["model_family"] == "segformer"
        assert r["encoder"] == "mit"
        assert r["train_mode"] == "frozen"

    def test_eomt(self):
        r = parse_run_name("EOMT_DINOv2_256x256_b4_frozen_e20_seed42")
        assert r["model_family"] == "eomt"
        assert r["encoder"] == "dinov2"
        assert r["train_mode"] == "frozen"

    def test_512_resolution(self):
        r = parse_run_name("UNET_VGG16_512x512_b2_aug1_rep1_ce_dice_frozen_e30_seed42")
        assert r["size_hw"] == (512, 512)

    def test_aug_repeats(self):
        r = parse_run_name("UNET_SCRATCH_256x256_b4_aug1_rep3_ce_e60_seed42")
        assert r["aug_repeats"] == 3
        assert r["loss_name"] == "ce"

    def test_no_aug(self):
        r = parse_run_name("UNET_SCRATCH_256x256_b4_aug0_rep1_ce_e60_seed42")
        assert r["aug"] is False

    def test_unknown_name(self):
        r = parse_run_name("some_random_run_name")
        assert r["model_family"] is None
        assert r["encoder"] is None


class TestExtractMetrics:

    def test_basic_extraction(self):
        hist = {
            "loss": [1.0, 0.8, 0.5, 0.3],
            "val_loss": [1.2, 0.9, 0.4, 0.35],
            "mIoU": [0.1, 0.2, 0.3, 0.4],
            "val_mIoU": [0.08, 0.18, 0.35, 0.38],
        }
        m = extract_metrics_from_history(hist)

        assert m["last_loss"] == pytest.approx(0.3, abs=1e-5)
        assert m["last_val_loss"] == pytest.approx(0.35, abs=1e-5)
        assert m["best_val_loss"] == pytest.approx(0.35, abs=1e-5)
        assert m["best_val_loss_epoch"] == 4
        assert m["last_mIoU"] == pytest.approx(0.4, abs=1e-5)
        assert m["best_val_mIoU"] == pytest.approx(0.38, abs=1e-5)
        assert m["best_val_mIoU_epoch"] == 4
        assert m["n_epochs_ran"] == 4

    def test_best_not_last(self):
        hist = {
            "val_loss": [1.0, 0.5, 0.3, 0.6, 0.8],
            "val_mIoU": [0.1, 0.3, 0.5, 0.4, 0.35],
        }
        m = extract_metrics_from_history(hist)

        assert m["best_val_loss_epoch"] == 3
        assert m["best_val_mIoU_epoch"] == 3

    def test_empty_history(self):
        assert extract_metrics_from_history({}) == {}
        assert extract_metrics_from_history(None) == {}

    def test_partial_history(self):
        hist = {"loss": [1.0, 0.5]}
        m = extract_metrics_from_history(hist)
        assert m["last_loss"] == pytest.approx(0.5, abs=1e-5)
        assert "last_val_loss" not in m
        assert "best_val_mIoU" not in m

    def test_lr_extraction(self):
        hist = {"loss": [1.0], "learning_rate": [0.001, 0.0005, 0.00025]}
        m = extract_metrics_from_history(hist)
        assert m["last_lr"] == pytest.approx(0.00025, abs=1e-7)

    def test_tail_stats(self):
        hist = {
            "val_mIoU": [0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.48, 0.47, 0.49, 0.50],
        }
        m = extract_metrics_from_history(hist)
        assert m["val_mIoU_tail_mean"] == pytest.approx(np.mean([0.45, 0.48, 0.47, 0.49, 0.50]), abs=1e-4)


class TestSafeFloat:
    def test_normal(self):
        assert safe_float(3.14) == pytest.approx(3.14)

    def test_int(self):
        assert safe_float(42) == 42.0

    def test_string_number(self):
        assert safe_float("2.5") == 2.5

    def test_none(self):
        assert np.isnan(safe_float(None))

    def test_bad_string(self):
        assert np.isnan(safe_float("not_a_number"))

    def test_empty_string(self):
        assert np.isnan(safe_float(""))


class TestRunDiscovery:

    def test_summary_json_roundtrip(self):
        summary = {
            "run_name": "UNET_VGG16_256x256_b4_aug1_rep1_ce_dice_frozen_e40_seed42",
            "val_loss": 0.45,
            "val_mIoU": 0.52,
            "test_loss": 0.48,
            "test_mIoU": 0.50,
            "train_time_sec": 3600.0,
            "params": {
                "model": "unet_vgg16",
                "encoder": "vgg16",
                "size_hw": [256, 256],
                "batch": 4,
                "epochs": 40,
                "aug": True,
                "aug_repeats": 1,
                "loss_name": "ce_dice",
                "trainable": False,
                "seed": 42,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(summary, f)
            f.flush()
            f_path = f.name

        try:
            loaded = json.loads(Path(f_path).read_text(encoding="utf-8"))
            assert loaded["run_name"] == summary["run_name"]
            assert loaded["val_mIoU"] == pytest.approx(0.52)
            assert loaded["params"]["model"] == "unet_vgg16"
        finally:
            os.unlink(f_path)

    def test_history_json_roundtrip(self):
        history = {
            "loss": [1.0, 0.7, 0.5],
            "val_loss": [1.1, 0.8, 0.6],
            "mIoU": [0.1, 0.3, 0.45],
            "val_mIoU": [0.08, 0.28, 0.42],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(history, f)
            f.flush()
            f_path = f.name

        try:
            loaded = json.loads(Path(f_path).read_text(encoding="utf-8"))
            assert len(loaded["loss"]) == 3
            m = extract_metrics_from_history(loaded)
            assert m["best_val_mIoU"] == pytest.approx(0.42, abs=1e-4)
        finally:
            os.unlink(f_path)

    def test_mock_experiment_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Run 1 : UNet VGG16
            run1 = exp_dir / "UNET_VGG16_256x256_b4_aug1_rep1_ce_dice_frozen_e40_seed42"
            run1.mkdir()
            (run1 / "summary.json").write_text(json.dumps({
                "val_mIoU": 0.52, "test_mIoU": 0.50,
                "train_time_sec": 1800,
            }))
            (run1 / "history.json").write_text(json.dumps({
                "loss": [1.0, 0.5], "val_loss": [1.1, 0.6],
                "mIoU": [0.2, 0.4], "val_mIoU": [0.15, 0.35],
            }))

            # Run 2 : SegFormer
            run2 = exp_dir / "SEGFORMER_MIT-B0_256x256_b4_frozen_e30_seed42"
            run2.mkdir()
            (run2 / "summary.json").write_text(json.dumps({
                "val_mIoU": 0.60, "test_mIoU": 0.58,
                "train_time_sec": 2400,
            }))

            # scan
            rows = []
            for rd in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                parsed = parse_run_name(rd.name)
                rows.append({"run_name": rd.name, **parsed})

            df = pd.DataFrame(rows)
            assert len(df) == 2
            assert df.iloc[0]["model_family"] in ("segformer", "unet")


class TestPalette:
    def test_palette_shape(self):
        from scripts.preprocessing import PALETTE, CATEGORY_NAMES, N_CLASSES
        assert PALETTE.shape == (8, 3)
        assert len(CATEGORY_NAMES) == N_CLASSES
        assert N_CLASSES == 8

    def test_palette_dtype(self):
        from scripts.preprocessing import PALETTE
        assert PALETTE.dtype == np.uint8

    def test_category_names(self):
        from scripts.preprocessing import CATEGORY_NAMES
        assert CATEGORY_NAMES[0] == "void"
        assert CATEGORY_NAMES[1] == "flat"
        assert CATEGORY_NAMES[6] == "human"
        assert CATEGORY_NAMES[7] == "vehicle"

    def test_lut_mapping(self):
        from scripts.preprocessing import LUT, IGNORE_LABEL
        for lid in range(6):
            assert LUT[lid] == 0, f"labelId {lid} should map to void (0)"
        for lid in [6, 7, 8, 9, 10]:
            assert LUT[lid] == 1, f"labelId {lid} should map to flat (1)"
        assert LUT[23] == 5
        assert LUT[24] == 6
        assert LUT[25] == 6
        for lid in range(26, 34):
            assert LUT[lid] == 7
        assert LUT[IGNORE_LABEL] == IGNORE_LABEL

    def test_colorize_groups(self):
        from scripts.preprocessing import colorize_groups, PALETTE
        mask = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 0, 0, 0],
            [7, 7, 7, 7],
        ], dtype=np.uint8)
        mask_pil = Image.fromarray(mask, mode="L")
        rgb = colorize_groups(mask_pil)
        rgb_np = np.array(rgb)
        assert rgb_np.shape == (4, 4, 3)
        np.testing.assert_array_equal(rgb_np[0, 0], PALETTE[0])
        np.testing.assert_array_equal(rgb_np[1, 3], PALETTE[7])


class TestImports:
    def test_import_config(self):
        from scripts import config
        assert hasattr(config, "EXP_DIR")

    def test_import_preprocessing(self):
        from scripts import preprocessing
        assert hasattr(preprocessing, "CATEGORY_NAMES")

    def test_import_augmentations(self):
        from scripts import augmentations
        assert hasattr(augmentations, "make_train_aug")

    def test_augmentation_pipeline(self):
        from scripts.augmentations import make_train_aug
        aug = make_train_aug()
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, 8, (256, 256), dtype=np.uint8)
        result = aug(image=img, mask=mask)
        assert result["image"].shape == (256, 256, 3)
        assert result["mask"].shape == (256, 256)
