import tensorflow as tf


def userDefinedModel():
    """
    """
    model = tf.keras.models.Sequential()
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
