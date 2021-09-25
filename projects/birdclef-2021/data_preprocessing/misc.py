import os
import shutil
import pandas as pd
import numpy as np
import keras
import librosa
import librosa.display
import time
import re
from PIL import Image
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.io import imread
from itertools import chain

from GLOBAL_VARS import RANDOM_SEED, SPEC_SHAPE, BATCH_SIZE, NUM_EPOCHS, NUM_TRAIN_FILES, NUM_VAL_FILES, LEVEL_NOISE, SAMPLE_RATE, SIGNAL_LENGTH, SPEC_SHAPE, FMIN, FMAX

numpy = 'datasets/train/numpy_3_full'

train_dir = 'datasets/train/mel_train/'
train_dir_col = 'datasets/train/mel_train_col/'

val_dir = 'datasets/train/mel_val/'
val_dir_col = 'datasets/train/mel_val_col/'

dest_dir = 'datasets/train/mel_collapsed_train/'
dest_dir_prev = 'datasets/train/collapsed_train_melspecs_full_prev'

counter = 0
file_counter = 3500
for root, dirs, files in os.walk(val_dir):

    for file in files:
        
        #label = re.split('_', file)[1]
        #label = re.split('.', file)[0]
        
        full_path = root + '/' + file
        
        #os.rename(full_path, label + '_' + str(file_counter) + '.png')
        #file_counter += 1

        shutil.copy(full_path, val_dir_col)

        counter += 1
print(counter)

# DONT FORGET TO ADD SOUNDSCAPE SPECTOGRAMS!!!!

# Labels to digits
TRAIN_META = pd.read_csv('datasets/train/0garbage/train_metadata.csv')
labels = TRAIN_META['primary_label'].unique()

class_to_int = {k: v for v, k in enumerate(labels)}

subdirs, dirs, files = os.walk(dest_dir).__next__()

i = 0
y_train_one_hot = np.zeros((len(files), 397))
for file in files:
    entries = file.split('.')[0]
    entries = entries.split('_')
    entries = entries[:-1]
    
    y_entry = np.zeros(397)
    for entry in entries:
        
        idx = class_to_int[entry]
        
        y_entry[idx] = 1
        
    y_train_one_hot[i] = y_entry
    
    i += 1


# subdirs, dirs, files = os.walk(dest_dir_prev).__next__()
# m = len(files)
# filenames = []
# labels = np.zeros((m, 1)) 
# filenames_counter = 0
# labels_counter = -1

# for subdir, dirs, files in os.walk(train_dir):

#     for file in files:
#         filenames.append(file)
#         labels[filenames_counter, 0] = labels_counter
#         filenames_counter += 1
#     labels_counter += 1

# saving the y_labels_one_hot array as a .npy file
np.save('y_labels_one_hot.npy', y_train_one_hot)

# saving the filename array as .npy file
np.save('filenames.npy', files)

filenames_shuffled, y_labels_one_hot_shuffled = shuffle(files, y_train_one_hot)
np.save('y_labels_one_hot_shuffled.npy', y_labels_one_hot_shuffled)
np.save('filenames_shuffled.npy', filenames_shuffled)

filenames_shuffled_numpy = np.array(filenames_shuffled)

X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
    filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=0.2, random_state=RANDOM_SEED)   

# You can save these files as well. As you will be using them later for training and validation of your model.
np.save('X_train_filenames.npy', X_train_filenames)
np.save('y_train.npy', y_train)

np.save('X_val_filenames.npy', X_val_filenames)
np.save('y_val.npy', y_val)



# # TRIM TRAIN SOUNDSCAPES TO 5 SEC
# soundscape_dir = 'datasets/train/0garbage/train_soundscapes/'

# soundscape_labels = pd.read_csv('datasets/train/0garbage/train_soundscape_labels.csv')

# soundscape_labels = soundscape_labels.drop(labels = 'site', axis=1)

# for subdir, dirs, files in os.walk(soundscape_dir):

#     for file in files:

#         file_split = file.split('_')

#         sound_label = soundscape_labels[soundscape_labels['audio_id'] == int(file_split[0])]

#         path = 'datasets/train/0garbage/train_soundscapes/' + file
        
#         audio, _ = librosa.load(path, sr=32000)

#         counter = 0
#         for i in range(0, len(audio), int(5 * 32000)):
            
#             split = audio[i:i + int(5 * 32000)]

#             # End of signal?S
#             if len(split) < int(5 * 32000):
#                 break

#             # Save split
#             split_dir = 'datasets/train/splitted_train_soundscape/'
#             if not os.path.exists(split_dir):
#                 os.makedirs(split_dir)

#             save_path = (split_dir + str(sound_label['seconds'].iloc[counter]) + '_' + str(sound_label['audio_id'].iloc[counter]) + '_' + sound_label['birds'].iloc[counter])
#             np.save(save_path, split)

#             counter += 1




# CREATE MELSPECS FROM TRAIN SOUNDSCAPES
soundscape_dir = 'datasets/train/nocalls/'

file_count = 0
for subdir, dirs, files in os.walk(soundscape_dir):

    for file in files:

        file_path = soundscape_dir + file
        audio = np.load(file_path)

        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=audio,
                                                sr=SAMPLE_RATE,
                                                n_fft=1024,
                                                hop_length=hop_length,
                                                n_mels=SPEC_SHAPE[0],
                                                fmin=FMIN,
                                                fmax=FMAX)
        
        # Save melspectogram as image file
        melspec_save_dir = 'datasets/train/mel_nocalls/'
        if not os.path.exists(melspec_save_dir):
            os.makedirs(melspec_save_dir)
        save_path = (melspec_save_dir + file.replace('.npy', ''))
        np.save(save_path, mel_spec)

        file_count += 1





# # Trim train_sounscapes (remove nocall and where 1 > labels)
# train_soundscapes = pd.read_csv('datasets/train/0garbage/train_soundscape_labels.csv')
# birds = train_soundscapes['birds'].value_counts()

# keys = list(birds.keys())
# remove_keys = []
# for key in keys:
#     if ' ' in key or key == 'nocall':
#         remove_keys.append(key)

# for string in remove_keys:
#     train_soundscapes  = train_soundscapes[train_soundscapes['birds'].map(lambda x: str(x)!=string)]
# print(train_soundscapes)

# # Remove soundscapes melspecs where nocall and 1 > labels
# melspec_soundscape_dir = 'datasets/train/collapsed_melspecs_soundscape/'
# removed_dir = 'datasets/train/0garbage/removed_melspecs_soundscape/'

# for subdir, dirs, files in os.walk(melspec_soundscape_dir):

#     for file in files:
        
#         if 'nocall' in file or ' ' in file:
#             shutil.move(melspec_soundscape_dir + file, removed_dir + file)

# for subdir, dirs, files in os.walk(dest_dir):
#     col_train = files
# notcol_train = []
# for subdir, dirs, files in os.walk(train_dir):
#     notcol_train.append(files)

# nocol_train = list(chain.from_iterable(notcol_train))

# files_for_deletion = []
# for file in notcol_train:
#     if file not in col_train:
#         files_for_deletion.append(file)
# print(files_for_deletion)

import keras
import numpy as np
from numpy.core.numeric import indices
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import random
import librosa
from skimage.io import imread
from sklearn.utils import shuffle
from keras import applications
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

from GLOBAL_VARS import RANDOM_SEED, SPEC_SHAPE, BATCH_SIZE, NUM_EPOCHS, NUM_TRAIN_FILES, NUM_VAL_FILES, LEVEL_NOISE, TRAIN_MELS, N_CLASSES, NOCALL_MELS, CLASS_TO_INT, VAL_MELS

tf.random.set_seed(RANDOM_SEED)

batch_x = TRAIN_MELS[0 * 12 : (0+1) * 12]
batch_y = np.zeros((BATCH_SIZE, N_CLASSES))

data_x = []
for file in batch_x:
    data_x.append(np.load('datasets/train/mel_train_col/' + file))
data_x = np.array(data_x)

for idx, file in enumerate(batch_x):
    file_1 = data_x[idx]
    y_entry = np.zeros(397)
    file_1_label = file.split('_')[0]
    file_1_label_idx = CLASS_TO_INT[file_1_label]
    
    
    indices_same = True
    while indices_same:
        idx2 = random.randint(0, len(TRAIN_MELS) - 1) # Second file
        idx3 = random.randint(0, len(TRAIN_MELS) - 1) # Third file
    
        indices_same = (TRAIN_MELS[idx2] == TRAIN_MELS[idx3] or file == TRAIN_MELS[idx2] or file == TRAIN_MELS[idx3])
        
    
    file_2 = np.load('datasets/train/mel_train_col/' + TRAIN_MELS[idx2])
    file_2 = np.roll(file_2, random.randint(SPEC_SHAPE[1] / 16, SPEC_SHAPE[1] / 2),axis=1)
    file_2_label = TRAIN_MELS[idx2].split('_')[0]
    file_2_label_idx = CLASS_TO_INT[file_2_label]
    
    file_3 = np.load('datasets/train/mel_train_col/' + TRAIN_MELS[idx3])
    file_3 = np.roll(file_3, random.randint(SPEC_SHAPE[1] / 16, SPEC_SHAPE[1] / 2),axis=1)
    file_3_label = TRAIN_MELS[idx3].split('_')[0]
    file_3_label_idx = CLASS_TO_INT[file_3_label]

    # NORMALIZE
    file_1 -= file_1.min()
    file_1 /= file_1.max()
    file_1 **= (random.random() * 1.2 + 0.7)
    
    r2 = random.random()
    r3 = random.random()
    
    if r2 < 0.7 and r3 > 0.35:    # 45.5% 2 classes
        # NORMALIZE
        file_2 -= file_2.min()
        file_2 /= file_2.max()
        file_2 **= (random.random() * 1.2 + 0.7)
        
        # if y_3 == 1:
        #   if file_1_max > file_2_max:
        #     y_1 = 1
        #     y_2 = 1 - (file_1_max - file_2_max)/(file_1_max + file_2_max)
        #   else:
        #     y_1 = 1 - (file_2_max - file_1_max)/(file_2_max + file_1_max)
        #     y_2 = 1
            
        y_entry[file_1_label_idx] = 1
        y_entry[file_2_label_idx] = 1
        
        file_1 += file_2
        
    elif r2 < 0.7 and r3 < 0.35:    # 24.5% 3 classes
        # NORMALIZE
        file_2 -= file_2.min()
        file_2 /= file_2.max()
        file_2 **= (random.random() * 1.2 + 0.7)
        
        # NORMALIZE
        file_3 -= file_3.min()
        file_3 /= file_3.max()
        file_3 **= (random.random() * 1.2 + 0.7)
        
        y_entry[file_1_label_idx] = 1
        
        y_entry[file_2_label_idx] = 1
        file_1 += file_2
        
        y_entry[file_3_label_idx] = 1
        file_1 += file_3
        
    else:
        y_entry[file_1_label_idx] = 1
        
    data_x[idx] = file_1
    batch_y[idx] = y_entry


# NOCALL ADDITION
nocall_batch_size = random.randint(0, BATCH_SIZE - 1)
idxs_data_x = np.random.choice(range(BATCH_SIZE), nocall_batch_size, replace=False)
for idx in idxs_data_x:
    idx_nocall = random.randint(0, len(NOCALL_MELS) - 1)
    nocall = np.load('datasets/train/mel_nocalls_col/' + NOCALL_MELS[idx_nocall])
    
    #NORMALIZE NOCALL
    nocall -= nocall.min()
    nocall /= nocall.max()
    nocall *= random.random() + 0.25
    nocall **= (random.random() * 1.2 + 0.7)
    
    data_x[idx] += nocall
    
# TO DECIBELS
for i in range(32):  
    data_x[i] = librosa.power_to_db(data_x[i], ref=np.max)
    data_x[i] = (data_x[i] + 80) / 80 

# Add white noise
if random.random()<0.8:
    for i in range(32):
        white_noise = (np.random.sample((SPEC_SHAPE[0], SPEC_SHAPE[1])).astype(np.float32) + 9) * data_x[i].mean() * LEVEL_NOISE * (np.random.sample() + 0.3)
        data_x[i] = data_x[i] + white_noise

# Add bandpass noise
if random.random()<0.7:
    for i in range(32):
        a = random.randint(0, SPEC_SHAPE[0]//2)
        b = random.randint(a + 20, SPEC_SHAPE[0])
        data_x[i, a:b, :] += (np.random.sample((b - a, SPEC_SHAPE[1])).astype(np.float32) + 9) * 0.05 * data_x[i].mean() * LEVEL_NOISE  * (np.random.sample() + 0.3)

# NORMALIZE
for i in range(data_x.shape[0] - 1):
    data_x[i] = data_x[i] - data_x[i].min()
    data_x[i] = data_x[i]/data_x[i].max() 

# Add 3 channels
rgb_batch = np.repeat(data_x[..., np.newaxis], 3, -1)