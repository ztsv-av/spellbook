import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

from tensorflow.keras.applications import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

from globalVariables import DROPOUT_RATES, FC_LAYERS

MODELS_CLASSIFICATION = {
    # 'DenseNet121': DenseNet121,
    'InceptionV3': InceptionV3}
    # 'Xception': Xception}
    # 'VGG16': VGG16}
    # 'MobileNet': MobileNet,
    # 'MobileNetV2': MobileNetV2,
    # 'ResNet50': ResNet50,
    # 'ResNet50V2': ResNet50V2,
    # 'InceptionResNetV2': InceptionResNetV2,
    # 'DenseNet169': DenseNet169,
    # 'ResNet101': ResNet101,
    # 'ResNet101V2': ResNet101V2,
    # 'EfficientNetB0': EfficientNetB0,
    # 'EfficientNetB1': EfficientNetB1,
    # 'EfficientNetB2': EfficientNetB2}
    # 'EfficientNetB3': EfficientNetB3,
    # 'EfficientNetB4': EfficientNetB4,
    # 'EfficientNetB5': EfficientNetB5}


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


def unfreezeModel(model, num_input_layers, batch_norm, num_layers):

    num_last_layers = 3 + num_input_layers

    if batch_norm:

        num_last_layers += 1

    if FC_LAYERS != None:

        for layer in FC_LAYERS:

            if layer != None:

                num_last_layers += 1
        
        for layer in DROPOUT_RATES:

            if layer != None:

                num_last_layers += 1
        
    num_layers_unfreeze = num_layers + num_last_layers

    for layer in model.layers[-num_layers_unfreeze + 1: -num_last_layers]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    return model


def buildClassificationPretrainedModel(model_path, pretrained_model, custom_objects, num_classes, activation, load_model):
    """
    XXX

    parameters
    ----------
        model_path : XXX
            XXX

        custom_objects : XXX
            XXX

            example:
            custom_objects = {
                'f1': f1,
                'categorical_focal_loss_fixed': categorical_focal_loss(alpha=[[.25, .25]], gamma=2)}

        num_classes : XXX
            XXX

        activation : XXX
            XXX

    returns
    -------
        model : XXX
            XXX
    """

    if load_model:
        pretrained_model = tf.keras.models.load_model(model_path, custom_objects)

    model = tf.keras.models.Sequential()
    # add all layers except last layer
    for layer in pretrained_model.layers[:-1]:
        model.add(layer)

    # add last classification layer
    model.add(tf.keras.layers.Dense(num_classes, activation=activation))

    return model


def buildClassificationImageNetModel(inputs, model_name, model_imagenet, pooling, dropout_connect_rate, do_batch_norm, initial_dropout, concat_features_before, concat_features_after, fc_layers, dropout_rates, num_classes, activation, trainable, do_predictions):
    """
    builds classification ImageNet model given image input shape, number of classes, pooling and activation layers

    parameters
    ----------
        model_imagenet : XXX
            XXX

        input_shape : XXX
            XXX

        pooling : XXX
            XXX

        num_classes : int
            number of classes in dataset

        activation : XXX
            XXX

    returns
    -------
        model : XXX
            XXX
    """

    if 'EfficientNet' in model_name:

        imagenet_model = model_imagenet(
            include_top=False, weights='imagenet', input_tensor=inputs[0], drop_connect_rate=dropout_connect_rate, pooling=pooling)
    else:

        imagenet_model = model_imagenet(
            include_top=False, weights='imagenet', input_tensor=inputs[0], pooling=pooling)

    if not trainable:

        imagenet_model.trainable = False

    feature_extractor = imagenet_model.output

    if do_batch_norm:

        feature_extractor = tf.keras.layers.BatchNormalization()(feature_extractor)

    if initial_dropout is not None:

        feature_extractor = tf.keras.layers.Dropout(initial_dropout)(feature_extractor)
    
    if concat_features_before:

        feature_layers = [layer for layer in inputs[1:]]
        feature_layers.insert(0, feature_extractor)

        concat_layer = tf.keras.layers.Concatenate(name='concat_features')(feature_layers)
    
    else:

        concat_layer = None

    if (fc_layers is not None) or (dropout_rates is not None):

        for fc, dropout in zip(fc_layers, dropout_rates):

            if fc == None:

                continue

            else:

                if concat_layer is not None:

                    feature_extractor = tf.keras.layers.Dense(fc, activation='relu')(concat_layer)

                    concat_layer = None

                else:

                    feature_extractor = tf.keras.layers.Dense(fc, activation='relu')(feature_extractor)

            if dropout == None:

                continue

            else:

                feature_extractor = tf.keras.layers.Dropout(dropout)(feature_extractor)

    if concat_features_after:

        feature_layers = [layer for layer in inputs[1:]]
        feature_layers.insert(0, feature_extractor)

        concat_layer = tf.keras.layers.Concatenate(name='concat_features')(feature_layers)
    
    else:

        concat_layer = None

    if do_predictions:

        if concat_layer is not None:

            predictions = tf.keras.layers.Dense(num_classes, activation=activation)(concat_layer)

        else:

            predictions = tf.keras.layers.Dense(num_classes, activation=activation)(feature_extractor)

        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        return model

    else:

        if concat_layer is not None:

            model = tf.keras.Model(inputs=inputs, outputs=concat_layer)

        else:

            model = tf.keras.Model(inputs=inputs, outputs=feature_extractor)

        return model


def buildDetectionModel(num_classes, checkpoint_path, config_path, dummy_shape):
    """
    #TODO : napiwi tyt description, pomesti kommenti v description kakie nado -- ostal'nie ydali
    """

    # Download the checkpoint and put it into models/research/object_detection/test_data/
    # wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz -O ./efficientdet_d4_1024x1024.tar.gz
    # tar -xf efficientdet_d4_1024x1024.tar.gz
    # mv efficientdet_d4_coco17_tpu-32/checkpoint models/research/object_detection/test_data/
    # tf.keras.backend.clear_session()

    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(
        config_path, config_override=None)

    # Read in the object stored at the key 'model' of the configs dictionary
    model_config = configs['model']

    # Modify the number of classes from its default
    model_config.ssd.num_classes = num_classes

    # Freeze batch normalization
    model_config.ssd.freeze_batchnorm = True

    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    tmp_box_predictor_checkpoint = tf.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads, _box_prediction_head=detection_model._box_predictor._box_prediction_head)

    tmp_model_checkpoint = tf.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor, _box_predictor=tmp_box_predictor_checkpoint)

    # Define a checkpoint
    checkpoint = tf.train.Checkpoint(model=tmp_model_checkpoint)

    # Restore the checkpoint to the checkpoint path
    checkpoint.restore(checkpoint_path)

    # Run a dummy image through the model so that variables are created
    # For the dummy image, you can declare a tensor of zeros that has a shape that the preprocess() method can accept (i.e. [batch, height, width, channels]).
    # use the detection model's `preprocess()` method and pass a dummy image
    dummy = tf.zeros(shape=dummy_shape)
    tmp_image, tmp_shapes = detection_model.preprocess(dummy)

    # run a prediction with the preprocessed image and shapes
    tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

    # postprocess the predictions into final detections
    tmp_detections = detection_model.postprocess(
        tmp_prediction_dict, tmp_shapes)

    tf.keras.backend.set_learning_phase(True)

    return detection_model


def buildAutoencoderPetfinder(
    num_image_features, num_features, image_feature_extractor_fc, autoencoder_fc):

    input_image_features_layer = tf.keras.layers.Input(shape=(num_image_features, ), name='input_image')
    input_features_layer = tf.keras.layers.Input(shape=(num_features,), name='input_features')

    image_features = tf.keras.layers.Dense(image_feature_extractor_fc, activation='relu', name='dense_image_features')(input_image_features_layer)

    concatenate_layer = tf.keras.layers.Concatenate(name='concat')([image_features, input_features_layer])

    fc1 = tf.keras.layers.Dense(autoencoder_fc[0], activation='relu', name='encoder_dense')(concatenate_layer)
    fc2 = tf.keras.layers.Dense(autoencoder_fc[1], activation='relu', name='bottleneck')(fc1)
    fc3 = tf.keras.layers.Dense(autoencoder_fc[2], activation='relu', name='decoder_dense')(fc2)

    predictions = tf.keras.layers.Dense(num_features, name='predict_features')(fc3)

    autoencoder = tf.keras.models.Model(inputs=[input_image_features_layer, input_features_layer], outputs=predictions)

    return autoencoder
