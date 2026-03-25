from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

from .models import unet_scratch, unet_vgg16, unet_resnet50, unet_convnext_tiny, segformer_mitb0
from .datagen import CityscapesSequence
from .augmentations import make_train_aug
from .losses_metrics import MeanIoUArgmax, dice_loss_sparse
from .preprocessing import N_CLASSES, IGNORE_LABEL, CATEGORY_NAMES, colorize_groups, overlay, PALETTE



def reset_between_runs(seed: int = 42):
    K.clear_session()
    gc.collect()
    tf.random.set_seed(seed)
    np.random.seed(seed)


def compile_model(model, loss_name: str = "ce", lr: float = 1e-3):
    if loss_name == "ce":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    elif loss_name == "ce_dice":
        def loss(y_true, y_pred):
            y_true_ = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
            valid   = tf.not_equal(y_true_, IGNORE_LABEL)
            y_safe  = tf.where(valid, y_true_, tf.zeros_like(y_true_))
            ce      = tf.keras.losses.sparse_categorical_crossentropy(y_safe, y_pred)
            ce      = tf.where(valid, ce, 0.0)
            denom   = tf.reduce_sum(tf.cast(valid, tf.float32)) + 1e-6
            ce      = tf.reduce_sum(ce) / denom
            return ce + 0.5 * dice_loss_sparse(y_true, y_pred, n_classes=N_CLASSES)
    else:
        raise ValueError(f"loss_name inconnu: {loss_name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[MeanIoUArgmax(num_classes=N_CLASSES, name="mIoU")],
    )
    return model


def _plot_history(history: dict, run_dir: Path):
    """Génère loss.png et miou.png dans run_dir."""
    epochs = range(1, len(history.get("loss", [])) + 1)

    if "loss" in history and "val_loss" in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history["loss"],     label="train loss", color="#2196F3")
        ax.plot(epochs, history["val_loss"], label="val loss",   color="#F44336")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Loss par epoch")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "loss.png", dpi=100)
        plt.close(fig)

    miou_key = "mIoU" if "mIoU" in history else "miou"
    vmiou_key = "val_mIoU" if "val_mIoU" in history else "val_miou"
    if miou_key in history and vmiou_key in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history[miou_key],     label="train mIoU", color="#4CAF50")
        ax.plot(epochs, history[vmiou_key], label="val mIoU",   color="#FF9800")
        ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU")
        ax.set_title("mIoU par epoch")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "miou.png", dpi=100)
        plt.close(fig)


def _plot_pred_grid(model, test_seq: CityscapesSequence, run_dir: Path, n: int = 4):
    try:
        X_batch, y_batch = test_seq[0]
        X_batch = X_batch[:n]
        y_batch = y_batch[:n]
        preds = model.predict(X_batch, verbose=0)
        pred_masks = np.argmax(preds, axis=-1).astype(np.uint8)
        y_masks    = y_batch[..., 0].astype(np.uint8)

        from PIL import Image as PILImage
        cols = 3
        rows = n
        W, H = 256, 256
        canvas = np.ones((rows * H, cols * W, 3), dtype=np.uint8) * 240

        for i in range(n):
            # image RGB
            img_np = (X_batch[i] * 255).clip(0, 255).astype(np.uint8)
            img_pil = PILImage.fromarray(img_np).resize((W, H))
            canvas[i*H:(i+1)*H, 0*W:1*W] = np.array(img_pil)

            # GT mask
            gt_pil = colorize_groups(PILImage.fromarray(y_masks[i], mode="L").resize((W, H), PILImage.NEAREST))
            canvas[i*H:(i+1)*H, 1*W:2*W] = np.array(gt_pil)

            # Pred mask
            pred_pil = colorize_groups(PILImage.fromarray(pred_masks[i], mode="L").resize((W, H), PILImage.NEAREST))
            canvas[i*H:(i+1)*H, 2*W:3*W] = np.array(pred_pil)

        fig, ax = plt.subplots(figsize=(12, 4 * n))
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title("Image  |  GT  |  Prédiction", fontsize=14, pad=10)
        fig.tight_layout()
        fig.savefig(run_dir / "pred_grid.png", dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] pred_grid non généré: {e}")


def _save_run_artifacts(
    run_dir: Path,
    summary: dict,
    history: dict,
    model,
    test_seq: CityscapesSequence,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2),  encoding="utf-8")

    # Keras model
    try:
        model.save(str(run_dir / "best.keras"))
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder best.keras: {e}")

    # Plots
    _plot_history(history, run_dir)
    _plot_pred_grid(model, test_seq, run_dir)

    print(f"\n Artefacts sauvegardés dans: {run_dir}")


def _make_sequences(df_idx, base_dir, size_hw, batch, aug, aug_repeats, seed):
    train_df = df_idx[df_idx["split_final"] == "train"].copy()
    val_df   = df_idx[df_idx["split_final"] == "val"].copy()
    test_df  = df_idx[df_idx["split_final"] == "test"].copy()

    train_aug = make_train_aug() if aug else None

    train_seq = CityscapesSequence(
        train_df, base_dir=base_dir, batch_size=batch,
        size_hw=size_hw, augment=train_aug, shuffle=True,
        seed=seed, aug_repeats=aug_repeats,
    )
    val_seq = CityscapesSequence(
        val_df, base_dir=base_dir, batch_size=batch,
        size_hw=size_hw, augment=None, shuffle=False,
        seed=seed, aug_repeats=1,
    )
    test_seq = CityscapesSequence(
        test_df, base_dir=base_dir, batch_size=batch,
        size_hw=size_hw, augment=None, shuffle=False,
        seed=seed, aug_repeats=1,
    )
    return train_seq, val_seq, test_seq


def _standard_callbacks(best_path: str, patience: int):
    return [
        callbacks.ModelCheckpoint(
            best_path, monitor="val_mIoU", mode="max", save_best_only=True,
        ),
        callbacks.EarlyStopping(
            monitor="val_mIoU", mode="max",
            patience=patience, restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_mIoU", mode="max",
            patience=3, factor=0.5, min_lr=1e-6,
        ),
    ]


def run_unet_scratch(
    df_idx: pd.DataFrame,
    base_dir,
    size_hw=(256, 256),
    batch: int = 4,
    epochs: int = 60,
    aug: bool = True,
    aug_repeats: int = 1,
    loss_name: str = "ce_dice",
    patience: int = 8,
    out_dir: str = "out/experiments",
    seed: int = 42,
):
    out_dir = Path(out_dir)
    reset_between_runs(seed)

    run_name = (
        f"UNET_SCRATCH_{size_hw[0]}x{size_hw[1]}"
        f"_b{batch}_aug{int(aug)}_rep{aug_repeats}_{loss_name}"
        f"_e{epochs}_seed{seed}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_seq, val_seq, test_seq = _make_sequences(
        df_idx, base_dir, size_hw, batch, aug, aug_repeats, seed
    )

    model = unet_scratch(input_shape=(size_hw[0], size_hw[1], 3), n_classes=N_CLASSES, base=32)
    model = compile_model(model, loss_name=loss_name, lr=1e-3)

    best_path = str(run_dir / "best.keras")
    cb = _standard_callbacks(best_path, patience)

    t0 = time.time()
    hist = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=cb, verbose=1)
    t_train = time.time() - t0

    best_model = tf.keras.models.load_model(
        best_path,
        custom_objects={"MeanIoUArgmax": MeanIoUArgmax, "dice_loss_sparse": dice_loss_sparse},
        compile=False,
    )
    best_model = compile_model(best_model, loss_name=loss_name, lr=1e-3)

    val_res  = best_model.evaluate(val_seq,  verbose=0)
    test_res = best_model.evaluate(test_seq, verbose=0)

    summary = {
        "run_name":        run_name,
        "val_loss":        float(val_res[0]),
        "val_mIoU":        float(val_res[1]),
        "test_loss":       float(test_res[0]),
        "test_mIoU":       float(test_res[1]),
        "train_time_sec":  float(t_train),
        "params": {
            "model": "unet_scratch", "encoder": "scratch",
            "size_hw": list(size_hw), "batch": batch,
            "epochs": epochs, "aug": aug,
            "aug_repeats": aug_repeats, "loss_name": loss_name,
            "trainable": True, "seed": seed,
        },
    }
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}

    _save_run_artifacts(run_dir, summary, history, best_model, test_seq)

    return {**summary, "best_path": best_path, "history": history, "run_dir": str(run_dir)}


def run_unet_vgg16(
    df_idx: pd.DataFrame,
    base_dir,
    size_hw=(256, 256),
    batch: int = 4,
    epochs: int = 40,
    aug: bool = True,
    aug_repeats: int = 1,
    loss_name: str = "ce_dice",
    patience: int = 8,
    out_dir: str = "out/experiments",
    seed: int = 42,
    trainable: bool = False,
):
    out_dir = Path(out_dir)
    reset_between_runs(seed)

    mode_str = "finetune" if trainable else "frozen"
    run_name = (
        f"UNET_VGG16_{size_hw[0]}x{size_hw[1]}"
        f"_b{batch}_aug{int(aug)}_rep{aug_repeats}_{loss_name}"
        f"_{mode_str}_e{epochs}_seed{seed}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_seq, val_seq, test_seq = _make_sequences(
        df_idx, base_dir, size_hw, batch, aug, aug_repeats, seed
    )

    model = unet_vgg16(
        input_shape=(size_hw[0], size_hw[1], 3),
        n_classes=N_CLASSES,
        encoder_weights="imagenet",
        trainable=trainable,
    )
    model = compile_model(model, loss_name=loss_name, lr=1e-3)

    best_path = str(run_dir / "best.keras")
    cb = _standard_callbacks(best_path, patience)

    t0 = time.time()
    hist = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=cb, verbose=1)
    t_train = time.time() - t0

    best_model = tf.keras.models.load_model(
        best_path,
        custom_objects={"MeanIoUArgmax": MeanIoUArgmax, "dice_loss_sparse": dice_loss_sparse},
        compile=False,
    )
    best_model = compile_model(best_model, loss_name=loss_name, lr=1e-3)

    val_res  = best_model.evaluate(val_seq,  verbose=0)
    test_res = best_model.evaluate(test_seq, verbose=0)

    summary = {
        "run_name":       run_name,
        "val_loss":       float(val_res[0]),
        "val_mIoU":       float(val_res[1]),
        "test_loss":      float(test_res[0]),
        "test_mIoU":      float(test_res[1]),
        "train_time_sec": float(t_train),
        "params": {
            "model": "unet_vgg16", "encoder": "vgg16",
            "size_hw": list(size_hw), "batch": batch,
            "epochs": epochs, "aug": aug,
            "aug_repeats": aug_repeats, "loss_name": loss_name,
            "trainable": trainable, "seed": seed,
        },
    }
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}

    _save_run_artifacts(run_dir, summary, history, best_model, test_seq)

    return {**summary, "best_path": best_path, "history": history, "run_dir": str(run_dir)}


def run_unet_resnet50(
    df_idx: pd.DataFrame,
    base_dir,
    size_hw=(256, 256),
    batch: int = 4,
    epochs: int = 40,
    aug: bool = True,
    aug_repeats: int = 1,
    loss_name: str = "ce_dice",
    patience: int = 8,
    out_dir: str = "out/experiments",
    seed: int = 42,
    trainable: bool = False,
):
    out_dir = Path(out_dir)
    reset_between_runs(seed)

    mode_str = "finetune" if trainable else "frozen"
    run_name = (
        f"UNET_RESNET50_{size_hw[0]}x{size_hw[1]}"
        f"_b{batch}_aug{int(aug)}_rep{aug_repeats}_{loss_name}"
        f"_{mode_str}_e{epochs}_seed{seed}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_seq, val_seq, test_seq = _make_sequences(
        df_idx, base_dir, size_hw, batch, aug, aug_repeats, seed
    )

    model = unet_resnet50(
        input_shape=(size_hw[0], size_hw[1], 3),
        n_classes=N_CLASSES,
        encoder_weights="imagenet",
        trainable=trainable,
    )
    model = compile_model(model, loss_name=loss_name, lr=1e-3)

    best_path = str(run_dir / "best.keras")
    cb = _standard_callbacks(best_path, patience)

    t0 = time.time()
    hist = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=cb, verbose=1)
    t_train = time.time() - t0

    best_model = tf.keras.models.load_model(
        best_path,
        custom_objects={
            "MeanIoUArgmax": MeanIoUArgmax,
            "dice_loss_sparse": dice_loss_sparse,
            "ResNet50Preprocess": tf.keras.layers.Layer,  # placeholder
        },
        compile=False,
    )
    best_model = compile_model(best_model, loss_name=loss_name, lr=1e-3)

    val_res  = best_model.evaluate(val_seq,  verbose=0)
    test_res = best_model.evaluate(test_seq, verbose=0)

    summary = {
        "run_name":       run_name,
        "val_loss":       float(val_res[0]),
        "val_mIoU":       float(val_res[1]),
        "test_loss":      float(test_res[0]),
        "test_mIoU":      float(test_res[1]),
        "train_time_sec": float(t_train),
        "params": {
            "model": "unet_resnet50", "encoder": "resnet50",
            "size_hw": list(size_hw), "batch": batch,
            "epochs": epochs, "aug": aug,
            "aug_repeats": aug_repeats, "loss_name": loss_name,
            "trainable": trainable, "seed": seed,
        },
    }
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}

    _save_run_artifacts(run_dir, summary, history, best_model, test_seq)

    return {**summary, "best_path": best_path, "history": history, "run_dir": str(run_dir)}


def run_unet_convnext(
    df_idx: pd.DataFrame,
    base_dir,
    size_hw=(256, 256),
    batch: int = 4,
    epochs: int = 40,
    aug: bool = True,
    aug_repeats: int = 1,
    loss_name: str = "ce_dice",
    patience: int = 8,
    out_dir: str = "out/experiments",
    seed: int = 42,
    trainable: bool = False,
):
    out_dir = Path(out_dir)
    reset_between_runs(seed)

    mode_str = "finetune" if trainable else "frozen"
    run_name = (
        f"UNET_CONVNEXT_TINY_{size_hw[0]}x{size_hw[1]}"
        f"_b{batch}_aug{int(aug)}_rep{aug_repeats}_{loss_name}"
        f"_{mode_str}_e{epochs}_seed{seed}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUN : {run_name}")
    print(f"{'='*60}\n")

    train_seq, val_seq, test_seq = _make_sequences(
        df_idx, base_dir, size_hw, batch, aug, aug_repeats, seed
    )

    model = unet_convnext_tiny(
        input_shape=(size_hw[0], size_hw[1], 3),
        n_classes=N_CLASSES,
        encoder_weights="imagenet",
        trainable=trainable,
    )
    model = compile_model(model, loss_name=loss_name, lr=1e-3)
    model.summary(print_fn=lambda x: print(x))

    best_path = str(run_dir / "best.keras")
    cb = _standard_callbacks(best_path, patience)

    t0 = time.time()
    hist = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=cb, verbose=1)
    t_train = time.time() - t0

    from .models import ConvNeXtPreprocess
    best_model = tf.keras.models.load_model(
        best_path,
        custom_objects={
            "MeanIoUArgmax": MeanIoUArgmax,
            "dice_loss_sparse": dice_loss_sparse,
            "ConvNeXtPreprocess": ConvNeXtPreprocess,
        },
        compile=False,
    )
    best_model = compile_model(best_model, loss_name=loss_name, lr=1e-3)

    val_res  = best_model.evaluate(val_seq,  verbose=0)
    test_res = best_model.evaluate(test_seq, verbose=0)

    print(f"\n▶ val   loss={val_res[0]:.4f}  mIoU={val_res[1]:.4f}")
    print(f"▶ test  loss={test_res[0]:.4f}  mIoU={test_res[1]:.4f}")
    print(f"▶ temps d'entraînement : {t_train/60:.1f} min")

    summary = {
        "run_name":       run_name,
        "val_loss":       float(val_res[0]),
        "val_mIoU":       float(val_res[1]),
        "test_loss":      float(test_res[0]),
        "test_mIoU":      float(test_res[1]),
        "train_time_sec": float(t_train),
        "params": {
            "model": "unet_convnext_tiny", "encoder": "convnext_tiny",
            "size_hw": list(size_hw), "batch": batch,
            "epochs": epochs, "aug": aug,
            "aug_repeats": aug_repeats, "loss_name": loss_name,
            "trainable": trainable, "seed": seed,
        },
    }
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}

    _save_run_artifacts(run_dir, summary, history, best_model, test_seq)

    return {**summary, "best_path": best_path, "history": history, "run_dir": str(run_dir)}



def run_segformer(
    df_idx: pd.DataFrame,
    base_dir,
    size_hw=(256, 256),
    batch: int = 4,
    epochs: int = 40,
    aug: bool = True,
    aug_repeats: int = 1,
    loss_name: str = "ce_dice",
    patience: int = 8,
    out_dir: str = "out/experiments",
    seed: int = 42,
    trainable: bool = False,
    encoder_preset: str = "mit_b0_cityscapes_1024",
    projection_filters: int = 256,
    lr: float = 6e-5,
):
    out_dir = Path(out_dir)
    reset_between_runs(seed)

    mode_str = "finetune" if trainable else "frozen"
    encoder_short = encoder_preset.replace("_cityscapes_1024", "").replace("_", "-")
    run_name = (
        f"SEGFORMER_{encoder_short.upper()}_{size_hw[0]}x{size_hw[1]}"
        f"_b{batch}_aug{int(aug)}_rep{aug_repeats}_{loss_name}"
        f"_{mode_str}_e{epochs}_seed{seed}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUN : {run_name}")
    print(f"{'='*60}\n")

    train_seq, val_seq, test_seq = _make_sequences(
        df_idx, base_dir, size_hw, batch, aug, aug_repeats, seed
    )

    model = segformer_mitb0(
        input_shape=(size_hw[0], size_hw[1], 3),
        n_classes=N_CLASSES,
        encoder_preset=encoder_preset,
        trainable=trainable,
        projection_filters=projection_filters,
    )
    model = compile_model(model, loss_name=loss_name, lr=lr)
    model.summary(print_fn=lambda x: print(x))

    best_path = str(run_dir / "best.keras")
    cb = _standard_callbacks(best_path, patience)

    t0 = time.time()
    hist = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=cb, verbose=1)
    t_train = time.time() - t0

    from .models import SegFormerPreprocess
    best_model = tf.keras.models.load_model(
        best_path,
        custom_objects={
            "MeanIoUArgmax": MeanIoUArgmax,
            "dice_loss_sparse": dice_loss_sparse,
            "SegFormerPreprocess": SegFormerPreprocess,
        },
        compile=False,
    )
    best_model = compile_model(best_model, loss_name=loss_name, lr=lr)

    val_res  = best_model.evaluate(val_seq,  verbose=0)
    test_res = best_model.evaluate(test_seq, verbose=0)

    print(f"\n▶ val   loss={val_res[0]:.4f}  mIoU={val_res[1]:.4f}")
    print(f"▶ test  loss={test_res[0]:.4f}  mIoU={test_res[1]:.4f}")
    print(f"▶ temps d'entraînement : {t_train/60:.1f} min")

    summary = {
        "run_name":       run_name,
        "val_loss":       float(val_res[0]),
        "val_mIoU":       float(val_res[1]),
        "test_loss":      float(test_res[0]),
        "test_mIoU":      float(test_res[1]),
        "train_time_sec": float(t_train),
        "params": {
            "model": "segformer", "encoder": encoder_short,
            "encoder_preset": encoder_preset,
            "projection_filters": projection_filters,
            "size_hw": list(size_hw), "batch": batch,
            "epochs": epochs, "aug": aug,
            "aug_repeats": aug_repeats, "loss_name": loss_name,
            "trainable": trainable, "seed": seed, "lr": lr,
        },
    }
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}

    _save_run_artifacts(run_dir, summary, history, best_model, test_seq)

    return {**summary, "best_path": best_path, "history": history, "run_dir": str(run_dir)}
