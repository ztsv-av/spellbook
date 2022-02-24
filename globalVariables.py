import numpy as np
import pandas as pd
import albumentations as A


# classification
NUM_EPOCHS = 50
START_EPOCH = 15
BATCH_SIZES = {
    'VGG16': 16,
    'VGG19': 16,
    'DenseNet121': 32,
    'DenseNet169': 32,
    'EfficientNetB0': 10,
    'EfficientNetB1': 8,
    'EfficientNetB2': 6,
    'EfficientNetB3': 6,
    'EfficientNetB4': 4,
    'EfficientNetB5': 4,
    'InceptionResNetV2': 12,
    'InceptionV3': 32,
    'MobileNet': 12,
    'MobileNetV2': 12,
    'ResNet50': 12,
    'ResNet50V2': 12,
    'ResNet101': 12,
    'ResNet101V2': 12,
    'Xception': 32,
    'convnext_base_384_in22ft1k': 10,
    'swin_base_patch4_window12_384_in22k': 10,
    'Swin': 6}
    
INPUT_SHAPE = (384, 384, 3)

LOAD_FEATURES = True
CONCAT_FEATURES_BEFORE = False
CONCAT_FEATURES_AFTER = True

MODEL_POOLING = 'avg'  # 'max', 'avg', None
DROP_CONNECT_RATE = 0.2  # 0.2 - default
INITIAL_DROPOUT = None
DO_BATCH_NORM = False
FC_LAYERS = (1024, None)
DROPOUT_RATES = (None, None)

DO_PREDICTIONS = True
OUTPUT_ACTIVATION = 'softmax' # 'sigmoid', 'softmax', 'relu', None
NUM_CLASSES = 15587 # 26
NUM_ADD_CLASSES = [26]

UNFREEZE = True
UNFREEZE_FULL = False
NUM_UNFREEZE_LAYERS = {
    'InceptionV3': 63}

LOAD_WEIGHTS = False
LOAD_MODEL = False

DATA_FILEPATHS = 'projects/happywhale-2022/data/data_numpy_384_flipped_idxs/' # 'projects/happywhale-2022/data/data_numpy_384_flipped_idxs/' 'projects/testing_animals/data/all_384_notfull/'
TRAIN_FILEPATHS = 'projects/happywhale-2022/data/data_numpy_384_flipped_idxs/' # 'projects/happywhale-2022/data/train_numpy_384_flipped_idxs/' 'projects/testing_animals/data/train_384/'
VAL_FILEPATHS = 'projects/happywhale-2022/data/val_numpy_384_flipped_idxs/' # 'projects/happywhale-2022/data/val_numpy_384_flipped_idxs/' 'projects/testing_animals/data/val_384/'
DO_VALIDATION = False
MAX_FILES_PER_PART = 1100
RANDOM_STATE = 1337

METADATA = None
ID_COLUMN = 'id'
TARGET_FEATURE_COLUMNS = ['individual_id']
ADD_FEATURES_COLUMNS = ['species']
FILENAME_UNDERSCORE = False
CREATE_ONEHOT = True
ONEHOT_IDX = 2
ONEHOT_IDXS_ADD = [3]

DO_KFOLD = False
NUM_FOLDS = 4

TRAINED_MODELS_PATH = ''
CLASSIFICATION_CHECKPOINT_PATH = 'projects/happywhale-2022/training/weights/convnext_base_384_in22ft1k/no-folds/15/savedModel/'

SAVE_TRAIN_INFO_DIR = 'projects/happywhale-2022/training/info/'
SAVE_TRAIN_WEIGHTS_DIR = 'projects/happywhale-2022/training/weights/'

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
LEARNING_RATE = 0.000500000023748725

LR_EXP = False
LR_DECAY_STEPS = 500
LR_DECAY_RATE = 0.95

LR_LADDER = True
LR_LADDER_STEP = 0.5
LR_LADDER_EPOCHS = 10

# losses
FROM_LOGITS = False # from_logits=True => no activation function
LABEL_SMOOTHING = None
LOSS_REDUCTION = None

# metrics
METRIC_TYPE = 'custom' # 'custom'
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
PERMUTATION_PROBABILITY_CLASSIFICATION = 1 / 8
PERMUTATIONS_CLASSIFICATION = [
    A.RandomGamma(gamma_limit=GAMMA_LIMIT,
                  p=PERMUTATION_PROBABILITY_CLASSIFICATION),
    # A.HorizontalFlip(p=PERMUTATION_PROBABILITY_CLASSIFICATION),
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
