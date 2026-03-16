import os
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

from .models import unet_scratch
from .datagen import CityscapesSequence
from .augmentations import make_train_aug
from .losses_metrics import MeanIoUArgmax, dice_loss_sparse
from .preprocessing import N_CLASSES
from .preprocessing import IGNORE_LABEL
from scripts.models import unet_convnext_tiny


def reset_between_runs(seed=42):
    K.clear_session()
    gc.collect()
    tf.random.set_seed(seed)
    np.random.seed(seed)


def compile_model(model, loss_name="ce", lr=1e-3):
    if loss_name == "ce":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    elif loss_name == "ce_dice":

        def loss(y_true, y_pred):
            # y_true: (B,H,W,1)
            y_true_ = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)  # (B,H,W)
            valid = tf.not_equal(y_true_, IGNORE_LABEL)

            # labels safe (ignore -> 0)
            y_safe = tf.where(valid, y_true_, tf.zeros_like(y_true_))

            # CE par pixel (B,H,W)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y_safe, y_pred)
            ce = tf.where(valid, ce, 0.0)

            denom = tf.reduce_sum(tf.cast(valid, tf.float32)) + 1e-6
            ce = tf.reduce_sum(ce) / denom

            # return ce + dice_loss_sparse(y_true, y_pred, n_classes=N_CLASSES)
            return ce + 0.5 * dice_loss_sparse(y_true, y_pred, n_classes=N_CLASSES)

    else:
        raise ValueError(f"loss_name inconnu: {loss_name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[MeanIoUArgmax(num_classes=N_CLASSES, name="mIoU")],
    )
    return model


def run_unet_scratch(
    df_idx: pd.DataFrame,
    base_dir,
    size_hw=(256, 256),
    batch=4,
    epochs=60,
    aug=True,
    aug_repeats=1,
    loss_name="ce_dice",
    patience=8,
    out_dir="out/experiments",
    seed=42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reset_between_runs(seed)

    train_df = df_idx[df_idx["split_final"] == "train"].copy()
    val_df = df_idx[df_idx["split_final"] == "val"].copy()
    test_df = df_idx[df_idx["split_final"] == "test"].copy()

    train_aug = make_train_aug() if aug else None

    train_seq = CityscapesSequence(
        train_df,
        base_dir=base_dir,
        batch_size=batch,
        size_hw=size_hw,
        augment=train_aug,
        shuffle=True,
        seed=seed,
        aug_repeats=aug_repeats,
    )
    val_seq = CityscapesSequence(
        val_df,
        base_dir=base_dir,
        batch_size=batch,
        size_hw=size_hw,
        augment=None,
        shuffle=False,
        seed=seed,
        aug_repeats=1,
    )
    test_seq = CityscapesSequence(
        test_df,
        base_dir=base_dir,
        batch_size=batch,
        size_hw=size_hw,
        augment=None,
        shuffle=False,
        seed=seed,
        aug_repeats=1,
    )

    model = unet_scratch(
        input_shape=(size_hw[0], size_hw[1], 3), n_classes=N_CLASSES, base=32
    )
    model = compile_model(model, loss_name=loss_name, lr=1e-3)

    run_name = (
        f"UNET_SCRATCH_{size_hw[0]}x{size_hw[1]}"
        f"_b{batch}_aug{int(aug)}_rep{aug_repeats}_{loss_name}"
        f"_e{epochs}_seed{seed}"
    )
    best_path = out_dir / f"{run_name}.keras"

    cb = [
        callbacks.ModelCheckpoint(
            str(best_path), monitor="val_mIoU", mode="max", save_best_only=True
        ),
        callbacks.EarlyStopping(
            monitor="val_mIoU", mode="max", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_mIoU", mode="max", patience=3, factor=0.5, min_lr=1e-5
        ),
    ]

    t0 = time.time()
    hist = model.fit(
        train_seq, validation_data=val_seq, epochs=epochs, callbacks=cb, verbose=1
    )
    t_train = time.time() - t0

    best_model = tf.keras.models.load_model(
        str(best_path),
        custom_objects={
            "MeanIoUArgmax": MeanIoUArgmax,
            "dice_loss_sparse": dice_loss_sparse,
        },
        compile=False,
    )
    best_model = compile_model(best_model, loss_name=loss_name, lr=1e-3)

    val_res = best_model.evaluate(val_seq, verbose=0)
    test_res = best_model.evaluate(test_seq, verbose=0)

    return {
        "run_name": run_name,
        "best_path": str(best_path),
        "train_time_sec": float(t_train),
        "val_loss": float(val_res[0]),
        "val_mIoU": float(val_res[1]),
        "test_loss": float(test_res[0]),
        "test_mIoU": float(test_res[1]),
        "history": hist.history,
    }
