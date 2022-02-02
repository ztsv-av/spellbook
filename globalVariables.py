import numpy as np
import pandas as pd
import albumentations as A


# classification
NUM_EPOCHS = 60
START_EPOCH = 0
BATCH_SIZES = {
    'VGG16': 16,
    'VGG19': 16,
    'DenseNet121': 16,
    'DenseNet169': 8,
    'EfficientNetB0': 12,
    'EfficientNetB1': 10,
    'EfficientNetB2': 8,
    'EfficientNetB3': 6,
    'EfficientNetB4': 4,
    'EfficientNetB5': 4,
    'InceptionResNetV2': 16,
    'InceptionV3': 16,
    'MobileNet': 16,
    'MobileNetV2': 16,
    'ResNet50': 16,
    'ResNet50V2': 16,
    'ResNet101': 16,
    'ResNet101V2': 16,
    'Xception': 16}
    
INPUT_SHAPE = (256, 256, 3)

LOAD_FEATURES = False
NUM_ADD_FEATURES = [2, 167, 3, 4, 7, 3] # [2, 3, 7, 167, 3, 4]
CONCAT_FEATURES_BEFORE = True
CONCAT_FEATURES_AFTER = False

MODEL_POOLING = 'avg'  # 'max', 'avg', None
DROP_CONNECT_RATE = 0.2  # 0.2 - default
INITIAL_DROPOUT = 0.2
DO_BATCH_NORM = False
FC_LAYERS = (512, 64)
DROPOUT_RATES = (0.2, None)

DO_PREDICTIONS = True
OUTPUT_ACTIVATION = None  # 'sigmoid', 'softmax', 'relu', None
NUM_CLASSES = 1

UNFREEZE = True
UNFREEZE_FULL = False
NUM_UNFREEZE_LAYERS = 63

LOAD_WEIGHTS = False
LOAD_MODEL = False

DATA_FILEPATHS = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed/'
TRAIN_FILEPATHS = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed-train/'
VAL_FILEPATHS = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed-val/'
DO_VALIDATION = True
MAX_FILES_PER_PART = 900
RANDOM_STATE = 1337

METADATA = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/age-full-last-predicted-breeds.csv')
ID_COLUMN = 'id'
TARGET_FEATURE_COLUMNS = ['popularity']
ADD_FEATURES_COLUMNS = ['type', 'breed', 'age']

DO_KFOLD = False
NUM_FOLDS = 5

TRAINED_MODELS_PATH = ''
CLASSIFICATION_CHECKPOINT_PATH = 'projects/petfinder/petfinder-previous/training/weights/InceptionV3/no-folds/8/savedModel/'

SAVE_TRAIN_INFO_DIR = 'projects/petfinder/petfinder-previous/training/info/'
SAVE_TRAIN_WEIGHTS_DIR = 'projects/petfinder/petfinder-previous/training/weights/'

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

CHECKPOINT_PATH = 'object_detection/models/research/object_detection/test_data/checkpoint_efficientdet_d0/ckpt-0'  # change only /checkpoint_.../
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

LR_EXP = False
LR_DECAY_STEPS = 500
LR_DECAY_RATE = 0.95

LR_LADDER = True
LR_LADDER_STEP = 0.5
LR_LADDER_EPOCHS = 15

# losses
FROM_LOGITS = True # from_logits=True => no activation function
LABEL_SMOOTHING = 0.005
LOSS_REDUCTION = None

# metrics
ACCURACY_THRESHOLD = 0.0  # use 0.0 when loss = BinaryCrossentropy(from_logits=True), otherwise 0.5 or any desired value
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

# denoising autoencoder
BUILD_AUTOENCODER = False
DENSE_NEURONS_DATA_FEATURES = 64
DENSE_NEURONS_ENCODER = 256
DENSE_NEURONS_BOTTLE = 64
DENSE_NEURONS_DECODER = 256

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
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2
HUE_LIMIT = 0
SATURATION_LIMIT = [15, 35]
VALUE_LIMIT = 0

DO_PERMUTATIONS = True
PERMUTATION_PROBABILITY_CLASSIFICATION = 1 / 4
PERMUTATIONS_CLASSIFICATION = [
    A.RandomGamma(gamma_limit=GAMMA_LIMIT,
                  p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.HorizontalFlip(p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    # A.GaussianBlur(blur_limit=GAUSSIAN_BLUR_LIMIT,
    #             p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.GlassBlur(max_delta=GLASS_BLUR_MAXDELTA, iterations=GLASS_BLUR_ITERATIONS,
                p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.Sharpen(alpha=SHARPEN_ALPHA, lightness=SHARPEN_LIGHTNESS,
            p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.Emboss(strength=EMBOSS_STRENGTH,
            p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.RandomBrightnessContrast(brightness_limit=BRIGHTNESS_LIMIT, 
        contrast_limit=CONTRAST_LIMIT, p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    A.HueSaturationValue(hue_shift_limit=HUE_LIMIT, sat_shift_limit=SATURATION_LIMIT, 
        val_shift_limit=VALUE_LIMIT, p=PERMUTATION_PROBABILITY_CLASSIFICATION)]
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
