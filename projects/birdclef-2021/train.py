import keras
import tensorflow as tf
from keras import applications
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import numpy as np
import os
import pandas as pd
import random
import librosa
from skimage.io import imread
from sklearn.utils import shuffle

from globalVariables import (NUM_EPOCHS, NUM_CLASSES, BATCH_SIZES, INPUT_SHAPE, TRAIN_FILEPATHS, VAL_FILEPATHS, ADDITIONAL_FILEPATH, RANDOM_STATE, SAMPLING_RATE, INDENT_SECONDS, SIGNAL_LENGTH, NOISE_LEVEL, FMIN, FMAX)

tf.random.set_seed(RANDOM_STATE)

# ONE-HOT
TRAIN_META = pd.read_csv('datasets/train/0garbage/train_metadata.csv')
LABELS = TRAIN_META['primary_label'].unique()
CLASS_TO_INT = {k: v for v, k in enumerate(LABELS)}
print(CLASS_TO_INT)


#strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) # FOR MULTI GPU TRAINING

class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, filenames, batch_size, path) :
    self.filenames = filenames
    self.batch_size = batch_size
    self.path = path
    
    
  def __len__(self) :
    return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = np.zeros((self.batch_size, NUM_CLASSES))
    
    data_x = []
    for file in batch_x:
      data_x.append(np.load(self.path + file))
    data_x = np.array(data_x)
    
    
    # Second and third class addition
    for idx, file in enumerate(batch_x):
      
      file_1 = data_x[idx]
      y_entry = np.zeros(397)
      file_1_label = file.split('_')[0]
      file_1_label_idx = CLASS_TO_INT[file_1_label]
      
      
      indices_same = True
      while indices_same:
        idx2 = random.randint(0, len(self.filenames) - 1) # Second file
        idx3 = random.randint(0, len(self.filenames) - 1) # Third file
        
        indices_same = (self.filenames[idx2] == self.filenames[idx3] or file == self.filenames[idx2] or file == self.filenames[idx3])
          
        
      file_2 = np.load(self.path + self.filenames[idx2])
      file_2 = np.roll(file_2, random.randint(INPUT_SHAPE[1] / 16, INPUT_SHAPE[1] / 2),axis=1)
      file_2_label = self.filenames[idx2].split('_')[0]
      file_2_label_idx = CLASS_TO_INT[file_2_label]
      
      file_3 = np.load(self.path + self.filenames[idx3])
      file_3 = np.roll(file_3, random.randint(INPUT_SHAPE[1] / 16, INPUT_SHAPE[1] / 2),axis=1)
      file_3_label = self.filenames[idx3].split('_')[0]
      file_3_label_idx = CLASS_TO_INT[file_3_label]
      
      
      # file_1_max, file_2_max, file_3_max = file_1.max(), file_2.max(), file_3.max()
      # file_max_idx = np.argmax([file_1_max, file_2_max, file_3_max])
      
      # if file_max_idx == 0:
      #   y_1 = 1
      #   y_2 = 1 - (file_1_max - file_2_max)/(file_1_max + file_2_max)
      #   y_3 = 1 - (file_1_max - file_3_max)/(file_1_max + file_3_max)
        
      # elif file_max_idx == 1:
      #   y_1 = 1 - (file_2_max - file_1_max)/(file_2_max + file_1_max)
      #   y_2 = 1
      #   y_3 = 1 - (file_2_max - file_3_max)/(file_2_max + file_3_max)
        
      # elif file_max_idx == 2:
      #   y_1 = 1 - (file_3_max - file_1_max)/(file_3_max + file_1_max)
      #   y_2 = 1 - (file_3_max - file_2_max)/(file_3_max + file_2_max)
      #   y_3 = 1
      
      # NORMALIZE
      if file_1.max() != file_1.min():
        file_1 -= file_1.min()
        file_1 /= file_1.max()
      file_1 **= (random.random() * 1.2 + 0.7)
      
      r2 = random.random()
      r3 = random.random()
      
      if r2 < 0.7 and r3 > 0.35:    # 45.5% 2 classes
        # NORMALIZE
        if file_2.max() != file_2.min():
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
        if file_2.max() != file_2.min():
          file_2 -= file_2.min()
          file_2 /= file_2.max()
        file_2 **= (random.random() * 1.2 + 0.7)
        
        # NORMALIZE
        if file_3.max() != file_3.min():
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
    nocall_batch_size = random.randint(0, data_x.shape[0] - 1)
    idxs_data_x = np.random.choice(range(data_x.shape[0] - 1), nocall_batch_size, replace=False)
    for idx in idxs_data_x:
      idx_nocall = random.randint(0, len(os.listdir(ADDITIONAL_FILEPATH)) - 1)
      nocall = np.load('datasets/train/mel_nocalls_col/' + ADDITIONAL_FILEPATH[idx_nocall])
      
      #NORMALIZE NOCALL
      if nocall.max() != nocall.min():
        nocall -= nocall.min()
        nocall /= nocall.max()
      nocall *= random.random() + 0.25
      nocall **= (random.random() * 1.2 + 0.7)
      
      data_x[idx] += nocall
      
     
    # TO DECIBELS
    for i in range(self.batch_size):  
      data_x[i] = librosa.power_to_db(data_x[i], ref=np.max)
      data_x[i] = (data_x[i] + 80) / 80 
    
    # Add white noise
    if random.random()<0.8:
      for i in range(self.batch_size):
        white_noise = (np.random.sample((INPUT_SHAPE[0], INPUT_SHAPE[1])).astype(np.float32) + 9) * data_x[i].mean() * NOISE_LEVEL * (np.random.sample() + 0.3)
        data_x[i] = data_x[i] + white_noise
    
    # Add bandpass noise
    if random.random()<0.7:
      for i in range(self.batch_size):
        a = random.randint(0, INPUT_SHAPE[0]//2)
        b = random.randint(a + 20, INPUT_SHAPE[0])
        data_x[i, a:b, :] += (np.random.sample((b - a, INPUT_SHAPE[1])).astype(np.float32) + 9) * 0.05 * data_x[i].mean() * NOISE_LEVEL  * (np.random.sample() + 0.3)
    
    
    # NORMALIZE
    for i in range(self.batch_size):
      data_x[i] = data_x[i] - data_x[i].min()
      data_x[i] = data_x[i]/data_x[i].max()
    
    # Add 3 channels
    rgb_batch = np.repeat(data_x[..., np.newaxis], 3, -1)

    return rgb_batch, np.array(batch_y)


def buildModel():

  dense_model = applications.DenseNet121(include_top=False, weights='imagenet', pooling='avg', input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))

  model = Sequential()
  model.add(dense_model)

  # Classification layer
  model.add(Dense(NUM_CLASSES))

  return model


model = buildModel()

optimizer = tf.keras.optimizers.Adam()
loss_bce = tf.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_bce, metrics=['accuracy'])

callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=4),
            tf.keras.callbacks.ModelCheckpoint(filepath='bird_last.h5', monitor='val_loss', verbose=1, save_best_only=True)]

x_train, x_val = TRAIN_FILEPATHS, VAL_FILEPATHS
for epoch in range(NUM_EPOCHS):
  
  x_train, x_val = shuffle(x_train), shuffle(x_val)
  
  my_training_batch_generator = My_Custom_Generator(x_train, BATCH_SIZES, TRAIN_FILEPATHS)
  my_validation_batch_generator = My_Custom_Generator(x_val, BATCH_SIZES, VAL_FILEPATHS)

  model.fit_generator(generator=my_training_batch_generator, steps_per_epoch = int(os.listdir(TRAIN_FILEPATHS) // BATCH_SIZES),
                    verbose = 1, validation_data = my_validation_batch_generator, validation_steps = int(os.listdir(VAL_FILEPATHS) // BATCH_SIZES), callbacks=callbacks)
