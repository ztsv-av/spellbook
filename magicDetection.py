from train import detectionTrain
from globalVariables import (BATCH_SIZE_DETECTION, NUM_EPOCHS_DETECTION, NUM_CLASSES_DETECTION, DUMMY_SHAPE_DETECTION,
                            TRAIN_FILEPATHS_DETECTION, TRAIN_META_DETECTION, BBOX_FORMAT, LABEL_ID_OFFSET,
                            PERMUTATIONS_DETECTION, LEARNING_RATE, LR_DECAY_STEPS, LR_DECAY_RATE, MODEL_NAME_DETECTION,
                            CONFIG_PATH, CHECKPOINT_PATH, SAVE_CHECKPOINT_DIR, SAVE_TRAIN_INFO_DIR_DETECTION)

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder


# Load config into dictionary
configs = config_util.get_configs_from_pipeline_file(
    CONFIG_PATH, config_override=None)

# read object stored at key 'model' of config dictionary
model_config = configs['model']

# modify default number of classes
model_config.ssd.num_classes = NUM_CLASSES_DETECTION

# freeze batch normalization
model_config.ssd.freeze_batchnorm = True

detection_model = model_builder.build(
    model_config=model_config, is_training=True)

tmp_box_predictor_checkpoint = tf.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    _box_prediction_head=detection_model._box_predictor._box_prediction_head)

tmp_model_checkpoint = tf.train.Checkpoint(
    _feature_extractor=detection_model._feature_extractor, _box_predictor=tmp_box_predictor_checkpoint)

# define checkpoint and restore checkpoint to checkpoint path
checkpoint = tf.train.Checkpoint(model=tmp_model_checkpoint)
checkpoint.restore(CHECKPOINT_PATH)

dummy = tf.zeros(shape=DUMMY_SHAPE_DETECTION)
tmp_image, tmp_shapes = detection_model.preprocess(dummy)

# run prediction with preprocessed image and shapes
tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

# postprocess predictions into final detections
tmp_detections = detection_model.postprocess(
    tmp_prediction_dict, tmp_shapes)

tf.keras.backend.set_learning_phase(True)

# define list that contains layers that to be fine-tuned in detection model
tmp_list = []
for v in detection_model.trainable_variables:
    if v.name.startswith('WeightSharedConvolutionalBoxPredictor'):
        tmp_list.append(v)
to_fine_tune = tmp_list


def detection():
    """
    main working function that starts the whole training process for object detection and classification tasks
    the process is as follows:
        - model, its config file and weights are loaded from Tensorflow Object Detection API database
        - config file is modified with variables from globalVariables.py
        - training checkpoint is loaded and model weights are restored
        - model is created and initialized
        - learning rate and optimizer are initialized and added to the model object
        - training function is called and all initialized variables are passed to it
    every action is based on the global variables initialized in globalVariables.py
    """

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE, decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    detectionTrain(
        NUM_EPOCHS_DETECTION, BATCH_SIZE_DETECTION, 
        NUM_CLASSES_DETECTION, LABEL_ID_OFFSET,
        TRAIN_FILEPATHS_DETECTION, BBOX_FORMAT, TRAIN_META_DETECTION, 
        PERMUTATIONS_DETECTION, None,
        detection_model, MODEL_NAME_DETECTION, 
        optimizer, to_fine_tune,
        SAVE_CHECKPOINT_DIR, SAVE_TRAIN_INFO_DIR_DETECTION)
