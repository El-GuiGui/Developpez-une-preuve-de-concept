import math
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import Sequence

from .preprocessing import load_rgb, load_mask_labelids, remap_to_groups


class CityscapesSequence(Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        base_dir,
        batch_size: int,
        size_hw,
        augment=None,
        shuffle: bool = True,
        seed: int = 42,
        aug_repeats: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.base_dir = str(base_dir) if base_dir is not None else ""
        self.batch_size = int(batch_size)
        self.size_hw = tuple(size_hw)
        self.augment = augment
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.aug_repeats = int(max(1, aug_repeats))

        self._pairs = [
            (i, r) for i in range(len(self.df)) for r in range(self.aug_repeats)
        ]
        self.idx = np.arange(len(self._pairs))

        if self.shuffle:
            np.random.RandomState(self.seed).shuffle(self.idx)

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def _resolve_path(self, row, col_abs, col_rel):
        if col_abs in row and isinstance(row[col_abs], str) and len(row[col_abs]) > 0:
            return row[col_abs]
        return f"{self.base_dir}/{row[col_rel]}"

    def __getitem__(self, i):
        batch_ids = self.idx[i * self.batch_size : (i + 1) * self.batch_size]
        H, W = self.size_hw
        imgs, masks = [], []

        for k in batch_ids:
            row_i, rep_i = self._pairs[int(k)]
            r = self.df.iloc[row_i]

            img_path = self._resolve_path(r, "image_path", "image_rel")
            mask_path = self._resolve_path(r, "mask_path", "mask_rel")

            img = load_rgb(img_path).resize((W, H), Image.BILINEAR)
            m = remap_to_groups(load_mask_labelids(mask_path)).resize(
                (W, H), Image.NEAREST
            )

            img_np = np.array(img)
            m_np = np.array(m, dtype=np.uint8)

            if self.augment is not None:
                np.random.seed(self.seed + row_i * 1000 + rep_i)
                out = self.augment(image=img_np, mask=m_np)
                img_np, m_np = out["image"], out["mask"]

            imgs.append(img_np.astype(np.float32) / 255.0)
            masks.append(m_np.astype(np.uint8))

        X = np.stack(imgs, axis=0)
        y = np.stack(masks, axis=0)[..., None]
        return X, y
