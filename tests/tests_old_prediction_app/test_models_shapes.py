def test_models_output_shapes():
    import tensorflow as tf
    from scripts.models import unet_scratch, unet_vgg16, unet_resnet50

    x = tf.zeros((1, 256, 256, 3), dtype=tf.float32)

    m1 = unet_scratch(input_shape=(256, 256, 3), n_classes=8, base=32)
    y1 = m1(x)
    assert y1.shape == (1, 256, 256, 8)

    m2 = unet_vgg16(
        input_shape=(256, 256, 3), n_classes=8, encoder_weights=None, trainable=False
    )
    y2 = m2(x)
    assert y2.shape == (1, 256, 256, 8)

    m3 = unet_resnet50(
        input_shape=(256, 256, 3), n_classes=8, encoder_weights=None, trainable=False
    )
    y3 = m3(x)
    assert y3.shape == (1, 256, 256, 8)
