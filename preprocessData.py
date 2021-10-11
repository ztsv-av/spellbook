import tensorflow as tf

import os
import numpy as np
import pandas as pd
from PIL import Image

from globalVariables import PERMUTATIONS_CLASSIFICATION, PERMUTATIONS_DETECTION
from helpers import loadFashionMNIST, visualizeImage_Box, save2NP
from preprocessFunctions import minMaxNormalizeNumpy, addColorChannels, resizeImageBbox, normalizeBBox
from permutationFunctions import classification_permutations, detection_permutations

LOAD_SIZE = (28, 28)
RESHAPE_SIZE = [32, 32]
NUM_CHANNELS = 3
FILES_DIR = 'datasets/fashion_mnist/data/'
SAVE_TRAIN_FILES_DIR = 'projects/testing/datasets/train/'
SAVE_TEST_FILES_DIR = 'projects/testing/datasets/test/'

TRAIN_FILES_DIR_DETECTION = 'datasets/robot_dataset/train/'
TEST_FILES_DIR_DETECTION = 'datasets/robot_dataset/test/'
TRAIN_META = pd.read_csv('datasets/robot_dataset/metas/train_annotations.csv')
TEST_META = pd.read_csv('datasets/robot_dataset/metas/test_annotations.csv')
SAVE_TRAIN_FILES_DIR_DETECTION = 'projects/testing_detection/datasets/train/'
SAVE_TEST_FILES_DIR_DETECTION = 'projects/testing_detection/datasets/test/'
SAVE_TRAIN_META_PATH = 'projects/testing_detection/datasets/metas/train_meta.csv'
SAVE_TEST_META_PATH = 'projects/testing_detection/datasets/metas/test_meta.csv'


def classification():

    train_data, test_data = loadFashionMNIST(FILES_DIR, LOAD_SIZE)
    x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]

    for idx, (image, label) in enumerate(zip(x_train, y_train)):

        image = np.asarray(image).astype(dtype='uint8')

        image = addColorChannels(image, NUM_CHANNELS)

        image = tf.image.resize(image, RESHAPE_SIZE)

        filename = str(idx) + '_' + str(label)
        save2NP(image, SAVE_TRAIN_FILES_DIR + filename)


    for idx, (image, label) in enumerate(zip(x_test, y_test)):

        image = np.asarray(image).astype(dtype='uint8')

        image = addColorChannels(image, NUM_CHANNELS)

        image = tf.image.resize(image, RESHAPE_SIZE)

        filename = str(idx) + '_' + str(label)
        save2NP(image, SAVE_TEST_FILES_DIR + filename)

def detection():

    train_files = os.listdir(TRAIN_FILES_DIR_DETECTION)
    test_files = os.listdir(TEST_FILES_DIR_DETECTION)

    annotations = pd.DataFrame(columns=['filename', 'bboxes'])

    for idx, filename in enumerate(train_files):
        
        image = Image.open(TRAIN_FILES_DIR_DETECTION + filename)
        image = np.asarray(image).astype(np.uint8)

        record = TRAIN_META[TRAIN_META['filename'] == filename]

        if record.empty:
            bbox = [np.asarray([0, 0, 1, 1, 0]).astype(np.float32)]
            
            image, bbox = resizeImageBbox(image, bbox, 512, 512, 'pascal_voc')
            bbox = [0, 0, 1, 1]
            bbox = [[float(x) for x in bbox]]

        else:
            bbox = [np.asarray([record['xmin'].values[0], record['ymin'].values[0], record['xmax'].values[0], record['ymax'].values[0], 1]).astype(np.float32)]

            image, bbox = resizeImageBbox(image, bbox, 512, 512, 'pascal_voc')
            bbox = [normalizeBBox(bbox[0][1], bbox[0][0], bbox[0][3], bbox[0][2], image.shape)]
        
        save2NP(image, SAVE_TRAIN_FILES_DIR_DETECTION + filename)
        annotations = annotations.append({'filename': filename, 'bboxes': bbox}, ignore_index=True)
    
    annotations.to_csv(path_or_buf=SAVE_TRAIN_META_PATH, index=False)

    annotations = pd.DataFrame(columns=['filename', 'bboxes'])

    for idx, filename in enumerate(test_files):
        
        image = Image.open(TEST_FILES_DIR_DETECTION + filename)
        image = np.asarray(image).astype(np.uint8)

        record = TEST_META[TEST_META['filename'] == filename]

        if record.empty:
            bbox = [np.asarray([0, 0, 1, 1, 0]).astype(np.float32)]
            
            image, bbox = resizeImageBbox(image, bbox, 512, 512, 'pascal_voc')
            bbox = [0, 0, 1, 1]
            bbox = [[float(x) for x in bbox]]

        else:
            bbox = [np.asarray([record['xmin'].values[0], record['ymin'].values[0], record['xmax'].values[0], record['ymax'].values[0], 1]).astype(np.float32)]

            image, bbox = resizeImageBbox(image, bbox, 512, 512, 'pascal_voc')
            bbox = [normalizeBBox(bbox[0][1], bbox[0][0], bbox[0][3], bbox[0][2], image.shape)]
        
        save2NP(image, SAVE_TEST_FILES_DIR_DETECTION + filename)
        annotations = annotations.append({'filename': filename, 'bboxes': bbox}, ignore_index=True)
    
    annotations.to_csv(path_or_buf=SAVE_TEST_META_PATH, index=False)

