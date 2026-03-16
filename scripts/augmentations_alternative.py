import albumentations as A

def make_train_aug():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=8, val_shift_limit=6, p=0.3),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),

            A.GaussNoise(std_range=(0.01, 0.02), mean_range=(0.0, 0.0), p=0.2),

            A.Affine(
                scale=(0.92, 1.08),
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-5, 5),
                interpolation=1,
                mask_interpolation=0,
                border_mode=0,
                fill=0,
                fill_mask=0,
                p=0.35,
            ),
        ]
    )