import numpy as np
import pandas as pd
import albumentations as A


# classification
NUM_EPOCHS = 35
START_EPOCH = 0
BATCH_SIZES = {
    'VGG16': 32,
    'VGG19': 32,
    'DenseNet121': 48,
    'DenseNet169': 48,
    'EfficientNetB0': 64,
    'EfficientNetB1': 52,
    'EfficientNetB2': 46,
    'EfficientNetB3': 42,
    'EfficientNetB4': 24,
    'EfficientNetB5': 24,
    'EfficientNetB6': 32,
    'EfficientNetB7': 32,
    'InceptionResNetV2': 64,
    'InceptionV3': 48,
    'MobileNet': 32,
    'MobileNetV2': 64,
    'ResNet50': 64,
    'ResNet50V2': 64,
    'ResNet101': 32,
    'ResNet101V2': 32,
    'Xception': 46,
    'convnext_tiny': 46,
    'convnext_small': 24,
    'swin_tiny_patch4_window7_224': 46,
    'vit_tiny_patch16_224_in21k': 64,
    'vit_small_patch16_224_in21k': 46,
    'swsl_resnet18': 64,
    'swsl_resnet50': 64,
    'resnet50_gn': 48,
    'convmixer_768_32': 16,
    'cait_xxs24_384': 48,
    'convnext_base_384_in22ft1k': 32,
    'resnetv2_50x1_bitm_in21k': 48}
    
INPUT_SHAPE = (224, 448, 3)

USE_TFIMM_MODELS = False
BUILD_ARC = False

LOAD_FEATURES = False
CONCAT_FEATURES_BEFORE = False
CONCAT_FEATURES_AFTER = False

IMAGENET_WEIGHTS = 'imagenet'
EFFNET_WEIGHTS = 'imagenet'

MODEL_POOLING = None  # 'max', 'avg', None
DROP_CONNECT_RATE = 0.2  # 0.2 - default
INITIAL_DROPOUT = None
DO_BATCH_NORM = False
FC_LAYERS = None
DROPOUT_RATES = None
GAP_IDXS = [-1, -5, -9, -14]
ARC_DROPOUT = 0.3
ARC_DENSE = 1024

DO_PREDICTIONS = False
OUTPUT_ACTIVATION = 'sigmoid' # 'sigmoid', 'softmax', 'relu', None
NUM_CLASSES = 22 # 152
NUM_ADD_CLASSES = None

UNFREEZE = True
UNFREEZE_FULL = True
UNFREEZE_PERCENT = None # 25
UNFREEZE_BATCHNORM = False
NUM_UNFREEZE_LAYERS = {
    'VGG16': None,
    'VGG19': None,
    'DenseNet121': None,
    'DenseNet169': None,
    'EfficientNetB0': None,
    'EfficientNetB1': None,
    'EfficientNetB2': None,
    'EfficientNetB3': None,
    'EfficientNetB4': None,
    'EfficientNetB5': 280,
    'EfficientNetB6': 333,
    'EfficientNetB7': 407,
    'InceptionResNetV2': None,
    'InceptionV3': 63,
    'MobileNet': None,
    'MobileNetV2': None,
    'ResNet50': None,
    'ResNet50V2': None,
    'ResNet101': None,
    'ResNet101V2': None,
    'Xception': None,
    'convnext_base_384_in22ft1k': None,
    'swin_base_patch4_window12_384_in22k': None,
    'Swin': None}

LOAD_WEIGHTS = False
LOAD_MODEL = False

DATA_FILEPATHS = 'projects/birdclef-2022/data/data_melspecs_22_less15/' # 'projects/birdclef-2022/data/data_melspecs/' 'projects/testing_animals/data/all_384_notfull/'
TRAIN_FILEPATHS = 'projects/birdclef-2022/data/train_22_less15/' # 'projects/birdclef-2022/data/train/' 'projects/testing_animals/data/train_384/'
VAL_FILEPATHS = 'projects/birdclef-2022/data/val_22_less15/' # 'projects/birdclef-2022/data/val/' # 'projects/testing_animals/data/val_384/'
DO_VALIDATION = True
VAL_SPLIT = 0.25
MAX_FILES_PER_PART = 2000
RANDOM_STATE = 1337

METADATA = None
ID_COLUMN = 'id'
TARGET_FEATURE_COLUMNS = ['label_idx']
ADD_FEATURES_COLUMNS = None
FILENAME_UNDERSCORE = False
CREATE_ONEHOT = True
CREATE_SPARSE = False
LABEL_IDX = 2
LABEL_IDXS_ADD = None

DO_KFOLD = False
NUM_FOLDS = None

TRAINED_MODELS_PATH = ''
CLASSIFICATION_CHECKPOINT_PATH = 'projects/birdclef-2022/training/weights/EfficientNetB2/no-folds/20/savedModel/'

SAVE_TRAIN_INFO_DIR = 'projects/birdclef-2022/training/info/'
SAVE_TRAIN_WEIGHTS_DIR = 'projects/birdclef-2022/training/weights/'

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

# layers
ARCMARGIN_S = 30
ARCMARGIN_M = 0.3

# optimizers
OPTIMIZER = 'Adam'
LEARNING_RATE = 0.001
MOMENTUM_VALUE = 0.8
NESTEROV = True

# losses
FROM_LOGITS = False # from_logits=True => no activation function
LABEL_SMOOTHING = None
LOSS_REDUCTION = None

# metrics
METRIC_TYPE = 'tensorflow' # 'custom'
ACCURACY_THRESHOLD = 0.5  # use 0.0 when loss = BinaryCrossentropy(from_logits=True), otherwise 0.5 or any desired value
F1_SCORE_AVERAGE = 'macro'
F1_SCORE_THRESHOLD = 0.5

# callbacks
CUSTOM_LRS_EPOCHS = None

LR_LADDER = False
LR_LADDER_STEP = 0.5
LR_LADDER_EPOCHS = 10

REDUCE_LR_PLATEAU = True
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MINIMAL_LR = 0.00001
REDUCE_LR_METRIC = 'val_loss'

LR_EXP = False
LR_DECAY_STEPS = 500
LR_DECAY_RATE = 0.95

LR_CUSTOM_DECAY = False
LR_START_DECAY   = 0.000001
LR_MAX_DECAY     = 0.000005  
LR_MIN_DECAY     = 0.000001
LR_RAMP_EP_DECAY = 4
LR_SUS_EP_DECAY  = 0
LR_VALUE_DECAY = 0.9

EARLY_STOPPING_PATIENCE = 5
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

# melspectograms
SAMPLING_RATE = 21952
INDENT_SECONDS = 21952
SIGNAL_LENGTH = 5  # seconds
HOP_LENGTH = 245
N_MELS = 224
N_FFT = 1536
WIN_LENGTH = 1536
FMIN = 300

# permutationFunctions
SIGNAL_AMPLIFICATION = 100
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

DO_PERMUTATIONS = False
PERMUTATION_PROBABILITY_CLASSIFICATION = 1 / 12
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
