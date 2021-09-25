import os
import pandas as pd

RANDOM_SEED = 1337

# TRIM, NOISE AND SPECTOGRAMS
INDENT_SEC = 32000
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5 # seconds
LEVEL_NOISE = 0.05
FMIN = 500
FMAX = 12500
SPEC_SHAPE = (256, 512) # height x width

# TRAINING
TRAIN_MELS = os.listdir('datasets/train/mel_train_col/')
VAL_MELS = os.listdir('datasets/train/mel_val_col/')
NOCALL_MELS = os.listdir('datasets/train/mel_nocalls_col/')

BATCH_SIZE = 12
NUM_EPOCHS = 50
NUM_TRAIN_FILES = len(TRAIN_MELS)
NUM_VAL_FILES = len(VAL_MELS)
N_CLASSES = 397

# ONE-HOT
TRAIN_META = pd.read_csv('datasets/train/0garbage/train_metadata.csv')
LABELS = TRAIN_META['primary_label'].unique()
CLASS_TO_INT = {k: v for v, k in enumerate(LABELS)}
print(CLASS_TO_INT)