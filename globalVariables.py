import numpy as np
import pandas as pd
import os
import albumentations as A


# classification
NUM_EPOCHS = 60
START_EPOCH = 0
INPUT_SHAPE = (512, 512, 3)
NUM_FEATURES = 12
INITIAL_DROPOUT = 0.5
FC_LAYERS = None
DROPOUT_RATES = None
DROP_CONNECT_RATE = 0.4  # 0.2 - default
OUTPUT_ACTIVATION = 'softmax'  # 'sigmoid', 'softmax', 'relu', None
NUM_CLASSES = 4
UNFREEZE = True
NUM_UNFREEZE_LAYERS = None
IMAGE_FEATURE_EXTRACTOR_FC = 64
AUTOENCODER_FC = (256, 64, 256)
LABELS = ""
LABEL_TO_INTEGER = ""
MODEL_POOLING = 'avg'  # 'max', 'avg', None

DATA_FILEPATHS = 'projects/petfinder/petfinder-previous/data/images-preprocessed/512/'
ADDITIONAL_DATA_FILEPATHS = ''
TRAIN_FILEPATHS = 'projects/petfinder/petfinder-previous/data/train/512/'
VAL_FILEPATHS = 'projects/petfinder/petfinder-previous/data/val/512/'
DO_KFOLD = True
NUM_FOLDS = 5
RANDOM_STATE = 1337
METADATA = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/preprocessed_metadata.csv')
ID_COLUMN = 'id'
FEATURE_COLUMN = 'Maturity'  # ['Type', 'Age', 'Breed', 'Gender', 'Color', 'Maturity', 'Fur', 'Health']
FULL_RECORD = False
MAX_FILES_PER_PART = 300
MAX_FILEPARTS_TRAIN = len(os.listdir(TRAIN_FILEPATHS)) // MAX_FILES_PER_PART
MAX_FILEPARTS_VAL = len(os.listdir(VAL_FILEPATHS)) // MAX_FILES_PER_PART
SHUFFLE_BUFFER_SIZE = 4096
TRAINED_MODELS_PATH = ''
TRAINED_MODELS_FILES = None  # os.listdir()
SAVE_TRAIN_INFO_DIR = 'projects/petfinder/petfinder-previous/training/info/512/'
SAVE_TRAIN_WEIGHTS_DIR = 'projects/petfinder/petfinder-previous/training/weights/512/'
LOAD_WEIGHTS = False
CLASSIFICATION_CHECKPOINT_PATH = 'projects/petfinder/petfinder-previous/training/weights/512/Xception/fold-4/30/weights.h5'

BATCH_SIZES = {
    'DenseNet121': 16,
    'DenseNet169': 8,
    'EfficientNetB0': 12,
    'EfficientNetB1': 10,
    'EfficientNetB2': 8,
    'EfficientNetB3': 6,
    'EfficientNetB4': 4,
    'EfficientNetB5': 4,
    'InceptionResNetV2': 16,
    'InceptionV3': 8,
    'MobileNet': 16,
    'MobileNetV2': 16,
    'ResNet50': 16,
    'ResNet50V2': 16,
    'ResNet101': 16,
    'ResNet101V2': 16,
    'Xception': 8}

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
    'Xception': 4}

# for 512x512 (some over memory)
BATCH_SIZES_512 = {
    'DenseNet121': 16,
    'DenseNet169': 16,
    'EfficientNetB0': 8,
    'EfficientNetB1': 16,
    'EfficientNetB2': 16,
    'EfficientNetB3': 16,
    'EfficientNetB4': 16,
    'EfficientNetB5': 16,
    'InceptionResNetV2': 16,
    'InceptionV3': 16,
    'MobileNet': 16,
    'MobileNetV2': 16,
    'ResNet50': 16,
    'ResNet50V2': 16,
    'ResNet101': 16,
    'ResNet101V2': 16,
    'Xception': 16}

# object detection
MODEL_NAME_DETECTION = 'effdet0'
NUM_EPOCHS_DETECTION = 2
BATCH_SIZE_DETECTION = 2
INPUT_SHAPE_DETECTION = (512, 512)
DUMMY_SHAPE_DETECTION = (1, 512, 512, 3)
NUM_CLASSES_DETECTION = 1
LABEL_ID_OFFSET = 1
IMAGE_TYPE = np.uint8
BBOX_FORMAT = 'albumentations'

# change only /checkpoint_.../
CHECKPOINT_PATH = 'object_detection/models/research/object_detection/test_data/checkpoint_efficientdet_d0/ckpt-0'
CONFIG_PATH = 'object_detection/models/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config'
TRAIN_FILEPATHS_DETECTION = 'projects/testing_detection/datasets/train/'
TRAIN_META_DETECTION = pd.read_csv(
    'projects/testing_detection/datasets/metas/train_meta.csv')
TEST_FILEPATHS_DETECTION = 'projects/testing_detection/datasets/test/'
TEST_META_DETECTION = pd.read_csv(
    'projects/testing_detection/datasets/metas/test_meta.csv')
SAVE_CHECKPOINT_DIR = 'projects/testing_detection/training/weights/'
SAVE_TRAIN_INFO_DIR_DETECTION = 'projects/testing_detection/training/csvs/'

# optimizers
LEARNING_RATE = 0.001
LR_DECAY_STEPS = 500
LR_DECAY_RATE = 0.95
LR_LADDER = True
LR_LADDER_STEP = 0.75
LR_LADDER_EPOCHS = 10
LR_EXP = False

# losses
FROM_LOGITS = False
LABEL_SMOOTING = 0.001

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
INDENT_SECONDS = 32000
SIGNAL_LENGTH = 5  # seconds
HOP_LENGTH = int(SIGNAL_LENGTH * SAMPLING_RATE / (INPUT_SHAPE[1] - 1))
N_FFT = 1536
FMIN = 500
FMAX = 12500

# permutationFunctions
NOISE_LEVEL = 0.05
WHITE_NOISE_PROBABILITY = 0.8
BANDPASS_NOISE_PROBABILITY = 0.7
DOWNSCALE_MIN = 0.1
DOWNSCALE_MAX = 0.3
GAUSSIAN_BLUR_LIMIT = (5, 9)
GLASS_BLUR_ITERATIONS = 1
GLASS_BLUR_MAXDELTA = 3
GAMMA_LIMIT = (96, 164)
EMBOSS_STRENGTH = (0.4, 1)
SHARPEN_ALPHA = (0.2, 0.7)
SHARPEN_LIGHTNESS = (0.2, 0.7)
OPTICAL_DISTORT_LIMIT = 0.4
OPTICAL_SHIFT_LIMIT = 0.5  # doesn't really do much
ROTATE_LIMIT = 30
INVERT_PROBABILITY = 0.5

DO_PERMUTATIONS = True
PERMUTATION_PROBABILITY_CLASSIFICATION = 1 / 4
PERMUTATIONS_CLASSIFICATION = [
    A.RandomGamma(gamma_limit=GAMMA_LIMIT,
                  p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.HorizontalFlip(p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    # A.GaussianBlur(blur_limit=GAUSSIAN_BLUR_LIMIT,
    #             p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    # A.GlassBlur(max_delta=GLASS_BLUR_MAXDELTA, iterations=GLASS_BLUR_ITERATIONS,
    #             p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.Sharpen(alpha=SHARPEN_ALPHA, lightness=SHARPEN_LIGHTNESS,
            p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.Emboss(strength=EMBOSS_STRENGTH,
            p=PERMUTATION_PROBABILITY_CLASSIFICATION)]
    # A.Downscale(scale_min=DOWNSCALE_MIN, scale_max=DOWNSCALE_MIN,
    #             p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    # A.GridDistortion(p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    # A.OpticalDistortion(distort_limit=OPTICAL_DISTORT_LIMIT,
    #                     shift_limit=OPTICAL_SHIFT_LIMIT, p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    # A.Rotate(limit=ROTATE_LIMIT, p=PERMUTATION_PROBABILITY_CLASSIFICATION)]

PERMUTATION_PROBABILITY_DETECTION = 1 / 4
PERMUTATIONS_DETECTION = [
    A.HorizontalFlip(p=PERMUTATION_PROBABILITY_DETECTION),
    A.GaussianBlur(blur_limit=GAUSSIAN_BLUR_LIMIT,
                   p=PERMUTATION_PROBABILITY_DETECTION),
    A.RandomGamma(gamma_limit=GAMMA_LIMIT, p=PERMUTATION_PROBABILITY_DETECTION)]
# A.Rotate(limit=ROTATE_LIMIT, p=PERMUTATION_PROBABILITY_DETECTION),

# preprocessData
RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256
