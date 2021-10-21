from train import detectionTrain
from globalVariables import (BATCH_SIZE_DETECTION, NUM_EPOCHS_DETECTION, NUM_CLASSES_DETECTION, DUMMY_SHAPE_DETECTION, TRAIN_FILEPATHS_DETECTION, TRAIN_META_DETECTION, BBOX_FORMAT,
                             LABEL_ID_OFFSET, LEARNING_RATE, LR_DECAY_RATE, MODEL_NAME_DETECTION, PERMUTATIONS_DETECTION, CONFIG_PATH, CHECKPOINT_PATH, SAVE_CHECKPOINT_DIR, SAVE_TRAINING_CSVS_DIR_DETECTION)

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

    lr_decay_steps = 100
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE, decay_steps=lr_decay_steps, decay_rate=LR_DECAY_RATE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    detectionTrain(
        BATCH_SIZE_DETECTION, NUM_EPOCHS_DETECTION, NUM_CLASSES_DETECTION, LABEL_ID_OFFSET,
        TRAIN_FILEPATHS_DETECTION, BBOX_FORMAT, TRAIN_META_DETECTION, PERMUTATIONS_DETECTION,
        None, detection_model, MODEL_NAME_DETECTION, optimizer, to_fine_tune,
        SAVE_CHECKPOINT_DIR, SAVE_TRAINING_CSVS_DIR_DETECTION)
