import tensorflow as tf
from .preprocessing import IGNORE_LABEL, N_CLASSES


def dice_loss_sparse(y_true, y_pred, n_classes=N_CLASSES, eps=1e-6):
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    valid = tf.not_equal(y_true, IGNORE_LABEL)
    y_true = tf.where(valid, y_true, tf.zeros_like(y_true))

    y_true_oh = tf.one_hot(y_true, depth=n_classes)
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    valid_f = tf.cast(valid, tf.float32)[..., None]
    y_true_oh = y_true_oh * valid_f
    y_pred = y_pred * valid_f

    inter = tf.reduce_sum(y_true_oh * y_pred, axis=[0, 1, 2])
    denom = tf.reduce_sum(y_true_oh + y_pred, axis=[0, 1, 2])
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - tf.reduce_mean(dice)


class MeanIoUArgmax(tf.keras.metrics.Metric):
    def __init__(self, num_classes=N_CLASSES, name="mIoU", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.miou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        valid = tf.not_equal(y_true, IGNORE_LABEL)
        y_true = tf.boolean_mask(y_true, valid)
        y_pred = tf.boolean_mask(y_pred, valid)
        return self.miou.update_state(y_true, y_pred)

    def result(self):
        return self.miou.result()

    def reset_state(self):
        self.miou.reset_state()
