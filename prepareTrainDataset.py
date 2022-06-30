from helpers import evaluateString, getLabelFromPath, getFeaturesFromPath, loadNumpy, createOneHotVector, createSparseValue
from permutationFunctions import classification_permutations, detection_permutations, whiteNoise, bandpassNoise
from preprocessFunctions import randomMelspecPower, spectrogramToDecibels, normalizeSpectogram, melspecMonoToColor
from globalVariables import BANDPASS_NOISE_PROBABILITY, INPUT_SHAPE, NOISE_LEVEL, WHITE_NOISE_PROBABILITY, SIGNAL_AMPLIFICATION

import librosa
import numpy as np
import random

import tensorflow as tf


def preprocessData(
    data, 
    permutations, do_permutations, normalization, 
    is_val, 
    bboxes, bbox_format, is_detection):
    '''
    permutes given data and applies a normalization function

    parameters
    ----------
        data : numpy array
            data to preprocess

        permutations : list
            list with permutation functions

        normalization : function
            normalization function

        bboxes : list
            list of bounding boxes

        bbox_format : string
            describes in which format bounding boxes are

        is_detection : boolean
            True if the object detection model is being trained

        is_val : boolean
            True if it is the validation stage

    returns
    -------
        (data, bboxes) : tuple of arrays
            preprocessed data and bounding boxes

        or

        data_tensor : tensor
            tensor of preprocessed data
    '''

    if is_detection:

        data, bboxes = detection_permutations(
            data, bboxes, bbox_format, permutations)

        return (data, bboxes)

    else:
        if not is_val:
            if do_permutations:
                data = classification_permutations(data, permutations)

        data = normalization(data)

        data_tensor = tf.convert_to_tensor(data)

        return data_tensor


def prepareClassificationDataset(
    batch_size, num_classes, num_add_classes, 
    filepaths, filepaths_part, 
    meta, id_column, feature_columns, add_features_columns, 
    filename_underscore, create_onehot, create_sparse, label_idx, label_idxs_add,  
    permutations, do_permutations, normalization, 
    strategy, is_val):
    """
    prepares data for training for classification/encoding-decoding tasks
    the process of preparing data for training/validation is as follows:
        - create empty lists for data, target labels and optionally additional features
        - for each path in filepaths:
            - load data and append it to data list
            - load target labels and append them to target labels list
            - optionally load additional features and append to the corresponding list
        - map all loaded data to the preprocessData function to permute and normalize it
        - create a tf.data.Dataset.from_tensor_slices object using preprocessed data and target labels + additional features lists
        - batch Dataset object
        - use strategy.experimental_distribute_dataset on the batched Dataset to distribute it across GPUs while training the model

    parameters
    ----------

        batch_size : integer
            number of training examples in one batch of data

        num_classes : integer
            number of classes

        num_add_classes : list
            contains integers representing number of classes in additional features
            
        filepaths : list
            full paths to all files
        
        filepaths_part : list
            full paths to training part of files

        meta : dataframe
            metadata, table containing ids of files, additional features and features to predict

        id_column : string
            name of the id column in the metadata

        feature_columns : list
            names of target feature columns

        add_features_columns : list
            names of additional feature columns to add as an input when training

        filename_underscore : boolean
            True if end of a filename has a class after an underscore

        create_onehot : boolean
            whether to use metadata and load one-hot vector from there
            or create a new one using a name of the file

        create_sparse : boolean
            used to create sparse label
        
        label_idx : int
            which idx to use when splitting filename path by underscore to create one-hot vector

        label_idxs_add : list
             contains indicies to use when splitting filename path by underscore to create one-hot vectors for additional features

        permutations : list
            list of data permutation functions

        do_permutations : boolen
            either to perfrom data permutations or not

        normalization : function
            normalization function to apply to data

        strategy : tf.distribute object
            TensorFlow API used in distributed training

        is_val : boolean
            shows whether it is a validation or training iteration

    returns
    -------

        data_dataset_dist : strategy.experimental_distribute_dataset object
            loaded, preprocessed, and batched data
    """

    data_list = []
    labels_list = [[] for num_targets in feature_columns]

    if add_features_columns is not None:
        add_features_list = [[] for num_features in add_features_columns]
    else:
        add_features_list = []
    
    # start_load_numpy_time = time.time()
    # print('Start Loading Numpy Files...', flush=True)

    for path in filepaths_part:

        data = loadNumpy(path)
        data_list.append(data)

        for feature_idx, feature_column in enumerate(feature_columns):

            if create_onehot:
                label = createOneHotVector(path, label_idx, num_classes)

            elif create_sparse:
                label = createSparseValue(path, label_idx)
            
            else:
                label = getFeaturesFromPath(path, meta, id_column, feature_column, filename_underscore)

                if type(label) == int or type(label) == float:
                    label = np.array(label, dtype=np.int32)

                else:

                    label = evaluateString(label)

            label_tensor = tf.convert_to_tensor(label, dtype=tf.float32)
            labels_list[feature_idx].append(label_tensor)

        if add_features_columns is not None:

            for feature_idx, add_feature_columns in enumerate(add_features_columns):

                if create_onehot:
                    add_feature = createOneHotVector(path, label_idxs_add[feature_idx], num_add_classes[feature_idx])
                
                elif create_sparse:
                    add_feature = createSparseValue(path, label_idxs_add[feature_idx])
                
                else:
                    add_feature = getFeaturesFromPath(path, meta, id_column, add_feature_columns, filename_underscore)

                    if type(add_feature) == int or type(add_feature) == float:
                        add_feature = np.array(add_feature, dtype=np.int32)

                    else:
                        add_feature = evaluateString(add_feature)
                
                add_feature_tensor = tf.convert_to_tensor(add_feature, dtype=tf.float32)
                add_features_list[feature_idx].append(add_feature_tensor)

    # end_load_numpy_time = time.time()
    # print('Finished loading Numpy Files. Time Passed: ' + str(end_load_numpy_time - start_load_numpy_time), flush=True)

    # start_mapping_data_time = time.time()
    # print('Start Mapping Data...', flush=True)

    data_map = map(lambda data: preprocessData(
        data, permutations, do_permutations, normalization, is_val, bboxes=None, bbox_format=None, is_detection=False), data_list)
    data_mapped_list = list(data_map)

    # end_mapping_data_time = time.time()
    # print('Finished Mapping Data. Time Passed: ' + str(end_mapping_data_time - start_mapping_data_time), flush=True)

    # start_formatting_data_time = time.time()
    # print('Start Formatting Data...', flush=True)

    concat_data = [data_mapped_list]
    for add_feature in add_features_list:
         concat_data.append(add_feature)
    concat_data = tuple(concat_data)
    if len(concat_data) == 1:
        concat_data = concat_data[0]

    labels_list = tuple(labels_list)
    if len(labels_list) == 1:
        labels_list = labels_list[0]

    # end_formatting_data_time = time.time()
    # print('Finished Formatting Data. Time Passed: ' + str(end_formatting_data_time - start_formatting_data_time), flush=True)

    # start_creating_dataset_time = time.time()
    # print('Start Creating Tensorflow Dataset...', flush=True)

    data_dataset = tf.data.Dataset.from_tensor_slices(
        (concat_data, labels_list))

    data_dataset = data_dataset.batch(batch_size)

    data_dataset_dist = strategy.experimental_distribute_dataset(
        data_dataset)

    # end_creating_dataset_time = time.time()
    # print('Finished Creating Tensorflow Dataset. Time Passed: ' + str(end_creating_dataset_time - start_creating_dataset_time), flush=True)

    return data_dataset_dist


def prepareBIRDCLEFDataset(
    batch_size, num_classes, num_add_classes, 
    filepaths, filepaths_part, 
    meta, id_column, feature_columns, add_features_columns, 
    filename_underscore, create_onehot, create_sparse, label_idx, label_idxs_add,  
    permutations, do_permutations, normalization, 
    strategy, is_val):
    """
    prepares data for training for BIRDCLEF competition
    the process of preparing data for training/validation is as follows:
        - create empty lists for data, target labels and optionally additional features
        - for each path in filepaths:
            - load data and append it to data list
            - load target labels and append them to target labels list
            - optionally load additional features and append to the corresponding list
        - map all loaded data to the preprocessData function to permute and normalize it
        - create a tf.data.Dataset.from_tensor_slices object using preprocessed data and target labels + additional features lists
        - batch Dataset object
        - use strategy.experimental_distribute_dataset on the batched Dataset to distribute it across GPUs while training the model

    parameters
    ----------

        batch_size : integer
            number of training examples in one batch of data

        num_classes : integer
            number of classes

        num_add_classes : list
            contains integers representing number of classes in additional features

        filepaths : list
            full paths to all files
        
        filepaths_part : list
            full paths to training part of files

        meta : dataframe
            metadata, table containing ids of files, additional features and features to predict

        id_column : string
            name of the id column in the metadata

        feature_columns : list
            names of target feature columns

        add_features_columns : list
            names of additional feature columns to add as an input when training

        filename_underscore : boolean
            True if end of a filename has a class after an underscore

        create_onehot : boolean
            whether to use metadata and load one-hot vector from there
            or create a new one using a name of the file

        create_sparse : boolean
            used to create sparse label
        
        label_idx : int
            which idx to use when splitting filename path by underscore to create one-hot vector

        label_idxs_add : list
             contains indicies to use when splitting filename path by underscore to create one-hot vectors for additional features

        permutations : list
            list of data permutation functions

        do_permutations : boolen
            either to perfrom data permutations or not

        normalization : function
            normalization function to apply to data

        strategy : tf.distribute object
            TensorFlow API used in distributed training

        is_val : boolean
            shows whether it is a validation or training iteration

    returns
    -------

        data_dataset_dist : strategy.experimental_distribute_dataset object
            loaded, preprocessed, and batched data
    """

    data_list = []
    labels_list = [[] for num_targets in feature_columns]

    if add_features_columns is not None:
        add_features_list = [[] for num_features in add_features_columns]
    else:
        add_features_list = []

    for i, path in enumerate(filepaths_part):

        data = loadNumpy(path)
        data = randomMelspecPower(data, 3, 0.5)
        data *= (random.random() * SIGNAL_AMPLIFICATION + 1)
        data_list.append(data)

        for feature_idx, feature_column in enumerate(feature_columns):

            if create_onehot:
                label = createOneHotVector(path, label_idx, num_classes)

            elif create_sparse:
                label = createSparseValue(path, label_idx)
            
            else:
                label = getFeaturesFromPath(path, meta, id_column, feature_column, filename_underscore)

                if type(label) == int or type(label) == float:
                    label = np.array(label, dtype=np.int32)

                else:

                    label = evaluateString(label)

            label_tensor = tf.convert_to_tensor(label, dtype=tf.float32)

        indices_same = True
        while indices_same:
            idx2 = random.randint(0, len(filepaths) - 1) # second file
            idx3 = random.randint(0, len(filepaths) - 1) # third file
            indices_same = (filepaths[idx2] == filepaths[idx3] or path == filepaths[idx2] or path == filepaths[idx3])

        r2 = random.random()
        r3 = random.random()

        if r2 < 0.7 and r3 > 0.35:  # 45.5% 2 classes

            data_2 = loadNumpy(filepaths[idx2])
            data_2 = np.roll(data_2, random.randint(int(INPUT_SHAPE[1] / 16), int(INPUT_SHAPE[1] / 2)), axis=1)
            data_2 = randomMelspecPower(data_2, 3, 0.5)
            data_2 *= (random.random() * SIGNAL_AMPLIFICATION + 1)
            data_list[i] += data_2

            data_2_label = createOneHotVector(filepaths[idx2], label_idx, num_classes)
            data_2_tensor = tf.convert_to_tensor(data_2_label, dtype=tf.float32)
            data_2_label_idx = np.argmax(data_2_tensor)
            if label_tensor[data_2_label_idx] != 1:
                label_tensor += data_2_tensor

        elif r2 < 0.7 and r3 < 0.35:    # 24.5% 3 classes

            data_2 = loadNumpy(filepaths[idx2])
            data_2 = np.roll(data_2, random.randint(int(INPUT_SHAPE[1] / 16), int(INPUT_SHAPE[1] / 2)), axis=1)
            data_2 = randomMelspecPower(data_2, 3, 0.5)
            data_2 *= (random.random() * SIGNAL_AMPLIFICATION + 1)
            data_list[i] += data_2

            data_2_label = createOneHotVector(filepaths[idx2], label_idx, num_classes)
            data_2_tensor = tf.convert_to_tensor(data_2_label, dtype=tf.float32)
            data_2_label_idx = np.argmax(data_2_tensor)
            if label_tensor[data_2_label_idx] != 1:
                label_tensor += data_2_tensor

            data_3 = loadNumpy(filepaths[idx3])
            data_3 = np.roll(data_3, random.randint(int(INPUT_SHAPE[1] / 16), int(INPUT_SHAPE[1] / 2)), axis=1)
            data_3 = randomMelspecPower(data_3, 3, 0.5)
            data_3 *= (random.random() * SIGNAL_AMPLIFICATION + 1)
            data_list[i] += data_3

            data_3_label = createOneHotVector(filepaths[idx3], label_idx, num_classes)
            data_3_tensor = tf.convert_to_tensor(data_3_label, dtype=tf.float32)
            data_3_label_idx = np.argmax(data_3_tensor)
            if label_tensor[data_3_label_idx] != 1:
                label_tensor += data_3_tensor
            
        labels_list[feature_idx].append(label_tensor)
        
    for i in range(len(filepaths_part)):  

        data_list[i] = spectrogramToDecibels(data_list[i])
        data_list[i] = normalizeSpectogram(data_list[i])

        data_list[i] = whiteNoise(data_list[i], INPUT_SHAPE, NOISE_LEVEL, WHITE_NOISE_PROBABILITY)
        data_list[i] = bandpassNoise(data_list[i], INPUT_SHAPE, NOISE_LEVEL, BANDPASS_NOISE_PROBABILITY)

        data_list[i] = randomMelspecPower(data_list[i], 2, 0.7)

        data_list[i] = melspecMonoToColor(data_list[i], INPUT_SHAPE, normalization)

        data_list[i] = tf.convert_to_tensor(data_list[i])

    concat_data = [data_list]
    for add_feature in add_features_list:
         concat_data.append(add_feature)
    concat_data = tuple(concat_data)
    if len(concat_data) == 1:
        concat_data = concat_data[0]

    labels_list = tuple(labels_list)
    if len(labels_list) == 1:
        labels_list = labels_list[0]

    data_dataset = tf.data.Dataset.from_tensor_slices(
        (concat_data, labels_list))

    data_dataset = data_dataset.batch(batch_size)

    data_dataset_dist = strategy.experimental_distribute_dataset(
        data_dataset)

    return data_dataset_dist



def prepareDetectionDataset(filepaths, bbox_format, meta, num_classes, label_id_offset, permutations, normalization, is_val):
    """
    prepares data for training for object detection tasks

    parameters
    ----------

        filepaths : list
            full paths to files

        bbox_format : string
            format of bounding boxes

        meta : dataframe
            metadata, table containing ids of files, additional features, features to predict and coordintates of bounding boxes

        num_classes : integer
            number of classes to predict

        label_id_offset : integer
            shifts all classes by a certain number of indices
            so that the model receives one-hot labels where non-background
            classes start counting at the zeroth index

        permutations : list
            list of data permutation functions

        normalization : function
            normalization function to apply to data

        is_val : boolean
            shows whether it is a validation or training iteration

    returns
    -------

        images_tensors : list
            list of preprocessed image tensors

        bboxes_tensors : list
            list of preprocessed bounding boxes tensors

        classes_one_hot_tensors : list
            list of one-hot label tensors
    """

    images_batch = []
    bboxes_batch = []

    for filepath in filepaths:

        filename = filepath.split('/')[-1]
        record = meta[meta['filename'] == filename]

        image = np.load(filepath)
        bboxes = evaluateString(record['bboxes'].values[0])

        image, bboxes = preprocessData(
            image, permutations, normalization, is_val, bboxes, bbox_format, is_detection=True)

        images_batch.append(image)
        bboxes_batch.append(np.array(bboxes))

    images_tensors = []
    bboxes_tensors = []
    classes_one_hot_tensors = []

    for (image, bbox) in zip(images_batch, bboxes_batch):

        images_tensors.append(
            tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0))

        bboxes_tensors.append(
            tf.convert_to_tensor(bbox, dtype=tf.float32))

        zero_indexed_classes = tf.convert_to_tensor(
            np.ones(shape=[bbox.shape[0]], dtype=np.int32) - label_id_offset)

        classes_one_hot_tensors.append(
            tf.one_hot(zero_indexed_classes, num_classes))

    return images_tensors, bboxes_tensors, classes_one_hot_tensors
