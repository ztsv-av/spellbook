import tensorflow as tf


def userDefinedModel(num_classes, activation):

    """
    """
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation=activation)
    ])

    return model


MODELS_CLASSIFICATION = {
    'EfficientNetB0': tf.keras.applications.EfficientNetB0,
    'EfficientNetB1': tf.keras.applications.EfficientNetB1,
    'EfficientNetB2': tf.keras.applications.EfficientNetB2,
    'EfficientNetB3': tf.keras.applications.EfficientNetB3,
    'DenseNet121': tf.keras.applications.DenseNet121,
    'DenseNet169': tf.keras.applications.DenseNet169,
    'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
    'InceptionV3': tf.keras.applications.InceptionV3,
    'MobileNet': tf.keras.applications.MobileNet,
    'MobileNetV2': tf.keras.applications.MobileNetV2,
    'ResNet50': tf.keras.applications.ResNet50,
    'ResNet50V2': tf.keras.applications.ResNet50V2,
    'ResNet101': tf.keras.applications.ResNet101,
    'ResNet101V2': tf.keras.applications.ResNet101V2,
    'Xception': tf.keras.applications.Xception,
    'EfficientNetB4': tf.keras.applications.EfficientNetB4,
    'EfficientNetB5': tf.keras.applications.EfficientNetB5
}
