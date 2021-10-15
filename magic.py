import tensorflow as tf
import tensorflow.keras.backend as K
from object_detection.utils import config_util
from object_detection.builders import model_builder

import numpy as np

from globalVariables import (
    BATCH_SIZES, NUM_EPOCHS, NUM_CLASSES, INPUT_SHAPE, TRAIN_FILEPATHS, VAL_FILEPATHS, 
    PERMUTATIONS_CLASSIFICATION, SHUFFLE_BUFFER_SIZE, OUTPUT_ACTIVATION, MODEL_POOLING, 
    LEARNING_RATE, LR_DECAY_RATE, SAVE_MODELS_DIR, SAVE_TRAINING_CSVS_DIR)
from globalVariables import (
    BATCH_SIZE_DETECTION, NUM_EPOCHS_DETECTION, NUM_CLASSES_DETECTION, DUMMY_SHAPE_DETECTION, 
    TRAIN_FILEPATHS_DETECTION, TRAIN_META_DETECTION, BBOX_FORMAT, LABEL_ID_OFFSET, MODEL_NAME_DETECTION, 
    PERMUTATIONS_DETECTION, CONFIG_PATH, CHECKPOINT_PATH, SAVE_CHECKPOINT_DIR, SAVE_TRAINING_CSVS_DIR_DETECTION)

from models import MODELS_CLASSIFICATION, userDefinedModel
from helpers import buildClassificationImageNetModel, buildDetectionModel, getPathsList, getLabelFromFilename
from train import classificationCustomTrain, detectionTrain
from preprocessFunctions import minMaxNormalizeNumpy


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy(
    devices=["GPU:0", "GPU:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


# def classification():

#     for model_name, model_imagenet in MODELS_CLASSIFICATION.items():

#         with strategy.scope():

#             # create model, loss, optimizer and metrics instances here
#             # reset model, optimizer (AND learning rate), loss and metrics for each iteration

#             classification_model = buildClassificationImageNetModel(
#                 model_imagenet, MODEL_POOLING, INPUT_SHAPE, NUM_CLASSES, OUTPUT_ACTIVATION)

#             loss = tf.keras.losses.SparseCategoricalCrossentropy()
#             learning_rate = LEARNING_RATE
#             optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#             accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#             metrics = [accuracy]

#         classification_model.compile(optimizer=optimizer,
#                                         loss=loss,
#                                         metrics=metrics)

#         # rename optimizer weights to train multiple models
#         with K.name_scope(classification_model.optimizer.__class__.__name__):
#             for i, var in enumerate(classification_model.optimizer.weights):
#                 name = 'variable{}'.format(i)
#                 classification_model.optimizer.weights[i] = tf.Variable(
#                     var, name=name)
        
#         preprocessing_dict = {'normalization': minMaxNormalizeNumpy, 'to_color': True, 'resize': True, 'permutations': PERMUTATIONS_CLASSIFICATION}
#         classificationTrain(model_name, classification_model, preprocessing_dict, strategy)

#         del classification_model
#         del loss
#         del learning_rate
#         del optimizer
#         del accuracy
#         del metrics

#         K.clear_session()


def сlassificationСustom():

    # load data
    train_paths_list = getPathsList(TRAIN_FILEPATHS)
    val_paths_list = getPathsList(VAL_FILEPATHS)

    train_images_list = []
    train_labels_list = []
    for path in train_paths_list:
        train_images_list.append(np.load(path))
        train_labels_list.append(getLabelFromFilename(path))
    
    val_images_list = []
    val_labels_list = []
    for path in val_paths_list:
        val_images_list.append(np.load(path))
        val_labels_list.append(getLabelFromFilename(path))

    for model_name, model_imagenet in MODELS_CLASSIFICATION.items():

        batch_size_per_replica = BATCH_SIZES[model_name]
        batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

        with strategy.scope():

            # create model, loss, optimizer and metrics instances here
            # reset model, optimizer (AND learning rate), loss and metrics for each iteration
            
            classification_model = userDefinedModel(NUM_CLASSES, OUTPUT_ACTIVATION)

            # classification_model = buildClassificationImageNetModel(
            #     model_imagenet, INPUT_SHAPE, MODEL_POOLING, NUM_CLASSES, OUTPUT_ACTIVATION)

            loss_object = tf.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

            def compute_total_loss(labels, predictions):
                per_gpu_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(
                    per_gpu_loss, global_batch_size=batch_size)

            val_loss = tf.keras.metrics.Mean(name='val_loss')


            lr_decay_steps = 1000
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=lr_decay_steps, decay_rate=LR_DECAY_RATE)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='train_accuracy')
            val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='val_accuracy')

            # rename optimizer weights to train multiple models
            with K.name_scope(optimizer.__class__.__name__):
                for i, var in enumerate(optimizer.weights):
                    name = 'variable{}'.format(i)
                    optimizer.weights[i] = tf.Variable(
                        var, name=name)
        
        classificationCustomTrain(
            batch_size, NUM_EPOCHS, (train_images_list, train_labels_list), (val_images_list, val_labels_list), 
            PERMUTATIONS_CLASSIFICATION, minMaxNormalizeNumpy, SHUFFLE_BUFFER_SIZE, classification_model, loss_object, 
            val_loss, compute_total_loss, optimizer, train_accuracy, val_accuracy, SAVE_TRAINING_CSVS_DIR, SAVE_MODELS_DIR, 
            model_name, strategy)

        print('Finished Training ' + model_name + '!')

        del batch_size_per_replica
        del batch_size
        del classification_model
        del loss_object
        del val_loss
        del learning_rate
        del optimizer
        del train_accuracy
        del val_accuracy

        K.clear_session()

        break


def detection():  

    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(
        CONFIG_PATH, config_override=None)

    # Read in the object stored at the key 'model' of the configs dictionary
    model_config = configs['model']

    # Modify the number of classes from its default
    model_config.ssd.num_classes = NUM_CLASSES_DETECTION

    # Freeze batch normalization
    model_config.ssd.freeze_batchnorm = True

    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    tmp_box_predictor_checkpoint = tf.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads, 
        _box_prediction_head=detection_model._box_predictor._box_prediction_head)

    tmp_model_checkpoint = tf.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor, _box_predictor=tmp_box_predictor_checkpoint)

    # Define a checkpoint
    checkpoint = tf.train.Checkpoint(model=tmp_model_checkpoint)

    # Restore the checkpoint to the checkpoint path
    checkpoint.restore(CHECKPOINT_PATH)

    # Run a dummy image through the model so that variables are created
    # For the dummy image, you can declare a tensor of zeros that has a shape that the preprocess() method can accept (i.e. [batch, height, width, channels]).
    # use the detection model's `preprocess()` method and pass a dummy image
    dummy = tf.zeros(shape=DUMMY_SHAPE_DETECTION)
    tmp_image, tmp_shapes = detection_model.preprocess(dummy)

    # run a prediction with the preprocessed image and shapes
    tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

    # postprocess the predictions into final detections
    tmp_detections = detection_model.postprocess(
        tmp_prediction_dict, tmp_shapes)

    tf.keras.backend.set_learning_phase(True)

    lr_decay_steps = 500
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=lr_decay_steps, decay_rate=LR_DECAY_RATE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # define a list that contains the layers that you wish to fine tune in the detection model
    tmp_list = []
    for v in detection_model.trainable_variables:
        if v.name.startswith('WeightSharedConvolutionalBoxPredictor'):
            tmp_list.append(v)
    to_fine_tune = tmp_list

    detectionTrain(
        BATCH_SIZE_DETECTION, NUM_EPOCHS_DETECTION, NUM_CLASSES_DETECTION, LABEL_ID_OFFSET,
        TRAIN_FILEPATHS_DETECTION, BBOX_FORMAT, TRAIN_META_DETECTION, PERMUTATIONS_DETECTION, 
        minMaxNormalizeNumpy, detection_model, MODEL_NAME_DETECTION, optimizer, to_fine_tune, 
        SAVE_CHECKPOINT_DIR, SAVE_TRAINING_CSVS_DIR_DETECTION)