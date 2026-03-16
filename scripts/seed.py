import os
import random
import numpy as np
import tensorflow as tf


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def reset_tf(seed: int = 42) -> None:
    tf.keras.backend.clear_session()
    seed_everything(seed)
