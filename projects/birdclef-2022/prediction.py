import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import random
from tqdm.notebook import tqdm
from pydub import AudioSegment

import tensorflow.keras.backend as K
import tensorflow as tf

os.chdir('V:\Git\spellbook')

from globalVariables import INDENT_SECONDS, SAMPLING_RATE, SIGNAL_LENGTH, HOP_LENGTH, N_FFT, N_MELS, INPUT_SHAPE, SIGNAL_AMPLIFICATION


def kerasNormalize(model_name):

    if model_name == 'VGG16':

        normalization_function = tf.keras.applications.vgg16.preprocess_input

    elif model_name == 'VGG19':

        normalization_function = tf.keras.applications.vgg19.preprocess_input
    
    elif model_name == 'InceptionV3':

        normalization_function = tf.keras.applications.inception_v3.preprocess_input
        
    elif model_name == 'Xception':

        normalization_function = tf.keras.applications.xception.preprocess_input
        
    elif model_name == 'MobileNet':

        normalization_function = tf.keras.applications.mobilenet.preprocess_input

    elif model_name == 'MobileNetV2':

        normalization_function = tf.keras.applications.mobilenet_v2.preprocess_input
    
    elif model_name == 'InceptionResNetV2':

        normalization_function = tf.keras.applications.inception_resnet_v2.preprocess_input
    
    elif (model_name == 'ResNet50') or (model_name == 'ResNet101'):

        normalization_function = tf.keras.applications.resnet50.preprocess_input
        
    elif (model_name == 'ResNet50V2') or (model_name == 'ResNet101V2'):

        normalization_function = tf.keras.applications.resnet_v2.preprocess_input

    elif 'DenseNet' in model_name:

        normalization_function = tf.keras.applications.densenet.preprocess_input

    elif 'EfficientNet' in model_name:

        normalization_function = None

    return normalization_function


def addColorChannels(x, num_channels):

    x = np.repeat(x[..., np.newaxis], num_channels, -1)

    return x


def spectrogramToDecibels(x):

    x = librosa.power_to_db(x.astype(np.float32), ref=np.max)

    return x


def normalizeSpectogram(x):

    x = (x + 80) / 80

    return x


def randomMelspecPower(data, power, c):

    data -= data.min()
    data /= (data.max() + K.epsilon())
    data **= (random.random() * power + c)
    
    return data    


def melspecMonoToColor(x:np.ndarray, input_shape, normalization):

    x = addColorChannels(x, input_shape[-1])
    v = (255 * x)
    if normalization is not None:
        v = normalization(v)

    return v


def preprocessAudio(path, normalization):

    mel_specs = []

    audio, rate = librosa.load(path, sr=SAMPLING_RATE, offset=None)

    step_5_sec = SIGNAL_LENGTH * INDENT_SECONDS

    for i in range(0, len(audio), int(step_5_sec)):

        split = audio[i:i + int(step_5_sec)]

        if len(split) < int(step_5_sec):

            break
        
        else:

            mel_spec = librosa.feature.melspectrogram(
                y=split, sr=SAMPLING_RATE, 
                n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            
            # mel_spec = randomMelspecPower(mel_spec, 3, 0.5) permut?
            # mel_spec *= (random.random() * SIGNAL_AMPLIFICATION + 1) permut?
            mel_spec = spectrogramToDecibels(mel_spec)
            mel_spec = normalizeSpectogram(mel_spec)
            # normalize to 0 - 1?
            mel_spec = melspecMonoToColor(mel_spec, INPUT_SHAPE, normalization)
            mel_spec = tf.convert_to_tensor(mel_spec)

            mel_specs.append(mel_spec)
    
    return mel_specs


def prediction(test_csv_path, test_dir):

    models = {}

    test_csv = pd.read_csv(test_csv_path)
    test_row_ids = test_csv['row_id']

    for row_id in test_row_ids:

        info = row_id.split('_')
        id = info[0] + '_' + info[1]
        label = info[2]
        second = info[3]

        file_path = test_dir + id
        mel_specs = preprocessAudio(file_path, normalization)
        



                


