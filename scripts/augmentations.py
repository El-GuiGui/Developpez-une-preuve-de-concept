import albumentations as A


def make_train_aug():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.12, contrast_limit=0.12, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=6, sat_shift_limit=8, val_shift_limit=6, p=0.3
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.08,
                rotate_limit=5,
                border_mode=0,
                p=0.35,
            ),
        ]
    )
