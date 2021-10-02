import tensorflow as tf
import numpy as np

from helpers import loadFashionMNIST, visualizeImage_Box, save2NP
from preprocessFunctions import minMaxNormalizeNumpy, addColorChannels

LOAD_SIZE = (28, 28)
RESHAPE_SIZE = [32, 32]
NUM_CHANNELS = 3
FILES_DIR = 'datasets/fashion_mnist/data/'
SAVE_FILES_DIR_TRAIN = 'projects/testing/datasets/train/'
SAVE_FILES_DIR_TEST = 'projects/testing/datasets/test/'

train_data, test_data = loadFashionMNIST(FILES_DIR, LOAD_SIZE)
x_train, y_train = train_data[0], train_data[1]
x_test, y_test = test_data[0], test_data[1]

for idx, (image, label) in enumerate(zip(x_train, y_train)):

    image = np.asarray(image).astype(dtype='float32')

    image = minMaxNormalizeNumpy(image)

    image = addColorChannels(image, NUM_CHANNELS)

    image = tf.image.resize(image, RESHAPE_SIZE)

    filename = str(idx) + '_' + str(label)
    save2NP(image, SAVE_FILES_DIR_TRAIN + filename)


for idx, (image, label) in enumerate(zip(x_test, y_test)):

    image = np.asarray(image).astype(dtype='float32')

    image = minMaxNormalizeNumpy(image)

    image = addColorChannels(image, NUM_CHANNELS)

    image = tf.image.resize(image, RESHAPE_SIZE)

    filename = str(idx) + '_' + str(label)
    save2NP(image, SAVE_FILES_DIR_TEST + filename)

