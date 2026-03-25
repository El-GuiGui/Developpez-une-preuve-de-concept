from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model
import tensorflow as tf


def conv_block(x, filters):
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def unet_scratch(input_shape=(256, 256, 3), n_classes=8, base=32):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, base)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, base * 2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, base * 4)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, base * 8)
    p4 = layers.MaxPooling2D()(c4)

    bn = conv_block(p4, base * 16)

    u4 = layers.UpSampling2D()(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base * 8)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base * 4)

    u2 = layers.UpSampling2D()(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base * 2)

    u1 = layers.UpSampling2D()(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base)

    outputs = layers.Conv2D(n_classes, 1, padding="same", activation="softmax")(c8)
    return Model(inputs, outputs, name="unet_scratch")


def unet_vgg16(
    input_shape=(256, 256, 3), n_classes=8, encoder_weights="imagenet", trainable=False
):
    base = tf.keras.applications.VGG16(
        include_top=False, weights=encoder_weights, input_shape=input_shape
    )
    base.trainable = trainable

    s1 = base.get_layer("block1_conv2").output
    s2 = base.get_layer("block2_conv2").output
    s3 = base.get_layer("block3_conv3").output
    s4 = base.get_layer("block4_conv3").output
    b = base.get_layer("block5_conv3").output

    def up(x, skip, f):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)
        return x

    x = up(b, s4, 512)
    x = up(x, s3, 256)
    x = up(x, s2, 128)
    x = up(x, s1, 64)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax", padding="same")(x)
    return Model(base.input, outputs, name="unet_vgg16")


@tf.keras.utils.register_keras_serializable(package="proj8")
class ResNet50Preprocess(layers.Layer):
    def call(self, inputs):
        x = inputs * 255.0
        return tf.keras.applications.resnet50.preprocess_input(x)


def unet_resnet50(
    input_shape=(256, 256, 3),
    n_classes=8,
    encoder_weights="imagenet",
    trainable=False,
):
    inputs = layers.Input(shape=input_shape, name="image_rgb_01")

    x = ResNet50Preprocess(name="resnet50_preprocess")(inputs)

    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=encoder_weights,
        input_tensor=x,
    )
    base.trainable = trainable

    s1 = base.get_layer("conv1_relu").output  # /2
    s2 = base.get_layer("conv2_block3_out").output  # /4
    s3 = base.get_layer("conv3_block4_out").output  # /8
    s4 = base.get_layer("conv4_block6_out").output  # /16
    b = base.get_layer("conv5_block3_out").output  # /32

    def up(x, skip, f):
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)
        return x

    x = up(b, s4, 512)  # /16
    x = up(x, s3, 256)  # /8
    x = up(x, s2, 128)  # /4
    x = up(x, s1, 64)  # /2

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax", padding="same")(x)
    return Model(inputs, outputs, name="unet_resnet50")


# Convnext


from tensorflow.keras import layers, Model
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="proj8")
class ConvNeXtPreprocess(layers.Layer):
    def call(self, inputs):
        return inputs * 255.0


def _pick_by_hw(base_model, wanted_hw):
    picked = {}
    for layer in base_model.layers:
        try:
            out = layer.output
        except Exception:
            continue
        if out is None or len(out.shape) != 4:
            continue
        h = int(out.shape[1])
        w = int(out.shape[2])
        if (h, w) in wanted_hw:
            picked[(h, w)] = out
    missing = [hw for hw in wanted_hw if hw not in picked]
    if missing:
        raise ValueError(f"ConvNeXt: features manquantes pour {missing}")
    return picked


def unet_convnext_tiny(
    input_shape=(256, 256, 3),
    n_classes=8,
    encoder_weights="imagenet",
    trainable=False,
):
    inputs = layers.Input(shape=input_shape, name="image_rgb_01")
    x255 = ConvNeXtPreprocess(name="convnext_preprocess")(inputs)

    base = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        include_preprocessing=True,
        weights=encoder_weights,
        input_shape=input_shape,
    )
    base.trainable = trainable

    H, W = input_shape[0], input_shape[1]
    wanted = {
        (H // 4, W // 4),
        (H // 8, W // 8),
        (H // 16, W // 16),
        (H // 32, W // 32),
    }
    picked = _pick_by_hw(base, wanted)

    s2 = picked[(H // 4, W // 4)]
    s3 = picked[(H // 8, W // 8)]
    s4 = picked[(H // 16, W // 16)]
    b = picked[(H // 32, W // 32)]

    feat_model = tf.keras.Model(base.input, [s2, s3, s4, b], name="convnext_feat")
    s2, s3, s4, b = feat_model(x255)

    def up(t, skip, f):
        t = layers.UpSampling2D((2, 2), interpolation="bilinear")(t)
        t = layers.Concatenate()([t, skip])
        t = conv_block(t, f)
        return t

    t = up(b, s4, 512)
    t = up(t, s3, 256)
    t = up(t, s2, 128)

    t = layers.UpSampling2D((2, 2), interpolation="bilinear")(t)
    t = conv_block(t, 64)

    t = layers.UpSampling2D((2, 2), interpolation="bilinear")(t)
    outputs = layers.Conv2D(n_classes, 1, activation="softmax", padding="same")(t)

    return Model(inputs, outputs, name="unet_convnext_tiny")


# SegFormer
import keras_hub

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


@tf.keras.utils.register_keras_serializable(package="proj8")
class SegFormerPreprocess(layers.Layer):
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        return (x - IMAGENET_MEAN) / IMAGENET_STD


def segformer_mitb0(
    input_shape=(256, 256, 3),
    n_classes=8,
    encoder_preset="mit_b1_cityscapes_1024",
    trainable=False,
    projection_filters=256,
):
    encoder = keras_hub.models.MiTBackbone.from_preset(
        encoder_preset,
        image_shape=input_shape,
    )
    encoder.trainable = bool(trainable)

    backbone = keras_hub.models.SegFormerBackbone(
        image_encoder=encoder,
        projection_filters=projection_filters,
    )

    segmenter = keras_hub.models.SegFormerImageSegmenter(
        backbone=backbone,
        num_classes=n_classes,
        preprocessor=None,
    )

    inp = layers.Input(shape=input_shape, name="image_rgb_01")
    x = SegFormerPreprocess(name="segformer_preprocess")(inp)

    y = segmenter(x)

    if (y.shape[1] is not None and y.shape[2] is not None and
        (y.shape[1] != input_shape[0] or y.shape[2] != input_shape[1])):
        y = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear")(y)

    y = layers.Softmax(axis=-1, name="probs")(y)

    return Model(inp, y, name="segformer_mitb0")