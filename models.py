from tensorflow.keras.applications import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

MODELS_CLASSIFICATION = {
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'EfficientNetB2': EfficientNetB2,
    'EfficientNetB3': EfficientNetB3,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'InceptionResNetV2': InceptionResNetV2,
    'InceptionV3': InceptionV3,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'ResNet50': ResNet50,
    'ResNet50V2': ResNet50V2,
    'ResNet101': ResNet101,
    'ResNet101V2': ResNet101V2,
    'Xception': Xception,
    'EfficientNetB4': EfficientNetB4,
    'EfficientNetB5': EfficientNetB5}


def userDefinedModel(num_classes, activation):
    """
    XXX

    parameters
    ----------
    num_classes : XXX
        XXX

    activation : XXX
        XXX

    returns
    -------
        model : XXX
            XXX
    """

    model = Sequential([
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation=activation)])

    return model
