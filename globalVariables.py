import os
import pandas as pd
import albumentations as A
import numpy as np


# classification
NUM_EPOCHS = 5
# for 256x256
BATCH_SIZES_256 = {
    'DenseNet121': 128,
    'EfficientNetB0': 16,
    'EfficientNetB1': 16,
    'EfficientNetB2': 8,
    'EfficientNetB3': 8,
    'EfficientNetB4': 4,
    'EfficientNetB5': 1,
    'InceptionResNetV2': 8,
    'InceptionV3': 16,
    'MobileNet': 16,
    'MobileNetV2': 16,
    'ResNet50': 16,
    'ResNet50V2': 16,
    'ResNet101': 8,
    'ResNet101V2': 8,
    'Xception': 4
}
# for 512x512 (over memory)
BATCH_SIZES_512 = {
    'DenseNet121': 4,
    'EfficientNetB0': 4,
    'EfficientNetB1': 4,
    'EfficientNetB2': 4,
    'EfficientNetB3': 2,
    'EfficientNetB4': 2,
    'EfficientNetB5': 2,
    'InceptionResNetV2': 4,
    'InceptionV3': 8,
    'MobileNet': 8,
    'MobileNetV2': 8,
    'ResNet50': 8,
    'ResNet50V2': 8,
    'ResNet101': 4,
    'ResNet101V2': 4,
    'Xception': 4
}
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
OUTPUT_ACTIVATION = 'softmax'  # 'sigmoid', 'softmax', None
TRAIN_FILES_PATH = ''
VAL_FILES_PATH = ''
TRAINED_MODELS_PATH = ''
TRAINED_MODELS_FILES = None # os.listdir()
SAVE_MODELS_DIR = ''
SAVE_TRAINING_CSVS_DIR = ''
LABELS = ""
LABEL_TO_INTEGER = ""
MODEL_POOLING = 'avg'
SHUFFLE_BUFFER_SIZE = 4096

# object detection
NUM_EPOCHS_DETECTION = 5
BATCH_SIZE_DETECTION = 2
INPUT_SHAPE_DETECTION = (512, 512)
DUMMY_SHAPE_DETECTION = (1, 512, 512, 3)
NUM_CLASSES_DETECTION = 1
LABEL_ID_OFFSET = 1
IMAGE_TYPE = np.uint8
CHECKPOINT_PATH = 'object_detection/models/research/object_detection/test_data/checkpoint_efficientdet_d0/ckpt-0' # change only /checkpoint_.../
CONFIG_PATH = 'object_detection/models/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config'
TRAIN_FILES_PATH_DETECTION = 'datasets/robot_dataset/train/'
TRAIN_META_DETECTION = pd.read_csv('datasets/robot_dataset/metas/train_annotations.csv')
CHECKPOINT_SAVE_DIR = 'object_detection/saved_checkpoints/effdet0/'

# optimizers
LEARNING_RATE = 0.001

# losses
LABEL_SMOOTING = 0.01

# metrics
# use 0.0 when loss = BinaryCrossentropy(from_logits=True), otherwise 0.5 or any desired value
BINARY_ACCURACY_THRESHOLD = 0.0
F1_SCORE_AVERAGE = 'macro'

# callbacks
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 7
OVERFITTING_THRESHOLD = 1.3

# helpers
NUM_CHANNELS = 3
NUMPY_SAVE_PATH = ''
PNG_SAVE_PATH = ''

# melspectogram finetune (birdclef 2020-21)
SAMPLING_RATE = 21952
SIGNAL_LENGTH = 5  # seconds
HOP_LENGTH = int(SIGNAL_LENGTH * SAMPLING_RATE / (INPUT_SHAPE[1] - 1))
N_FFT = 1536
FMIN = 500
FMAX = 12500

# dataPermutation
NOISE_LEVEL = 0.05
WHITE_NOISE_PROBABILITY = 0.8
BANDPASS_NOISE_PROBABILITY = 0.7
DOWNSCALE_MIN = 0.3
DOWNSCALE_MAX = 0.6
GAUSSIAN_BLUR_LIMIT = (5, 11)
GLASS_BLUR_ITERATIONS = 1
GLASS_BLUR_MAXDELTA = 1
GAMMA_LIMIT = (175, 220)
EMBOSS_STRENGTH = (0.5, 0.9)
SHARPEN_ALPHA = (0.2, 0.4)
SHARPEN_LIGHTNESS = (0.5, 0.7)
OPTICAL_DISTORT_LIMIT = 0.4
OPTICAL_SHIFT_LIMIT = 0.5  # doesn't really do much
ROTATE_LIMIT = 30
INVERT_PROBABILITY = 0.5
BBOX_FORMAT = ''
PERMUTATION_PROBABILITY_CLASSIFICATION = 1 / 9
PERMUTATIONS_CLASSIFICATION = [
    A.GaussianBlur(blur_limit=GAUSSIAN_BLUR_LIMIT,
                   p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.GlassBlur(max_delta=GLASS_BLUR_MAXDELTA, iterations=GLASS_BLUR_ITERATIONS,
                p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.RandomGamma(gamma_limit=GAMMA_LIMIT,
                  p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.Sharpen(alpha=SHARPEN_ALPHA, lightness=SHARPEN_LIGHTNESS,
              p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.Downscale(scale_min=DOWNSCALE_MIN, scale_max=DOWNSCALE_MIN,
                p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.Emboss(strength=EMBOSS_STRENGTH,
             p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.GridDistortion(p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.OpticalDistortion(distort_limit=OPTICAL_DISTORT_LIMIT,
                        shift_limit=OPTICAL_SHIFT_LIMIT, p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.Rotate(limit=ROTATE_LIMIT, p=PERMUTATION_PROBABILITY_CLASSIFICATION)]
PERMUTATION_PROBABILITY_DETECTION = 1 / 4
PERMUTATIONS_DETECTION = [
    A.HorizontalFlip(p=PERMUTATION_PROBABILITY_DETECTION),
    A.Rotate(limit=ROTATE_LIMIT, p=PERMUTATION_PROBABILITY_DETECTION),
    A.GaussianBlur(blur_limit=GAUSSIAN_BLUR_LIMIT, p=PERMUTATION_PROBABILITY_DETECTION),
    A.RandomGamma(gamma_limit=GAMMA_LIMIT, p=PERMUTATION_PROBABILITY_DETECTION)]

# dataPreprocessing
RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256
