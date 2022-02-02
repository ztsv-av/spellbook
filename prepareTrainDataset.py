from pandas import concat
from helpers import evaluateString, getLabelFromPath, getFeaturesFromPath, loadNumpy, saveNumpyArray
from permutationFunctions import classification_permutations, detection_permutations

import numpy as np
import time
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
    batch_size, filepaths, 
    meta, id_column, feature_columns, add_features_columns,
    permutations, do_permutations, normalization, 
    strategy, is_val):
    """
    prepares data for training for classification/encoding-decoding tasks

    parameters
    ----------

        batch_size : integer
            the number of training examples in one batch of data

        filepaths : list
            full paths to files

        meta : dataframe
            metadata, table containing ids of files, additional features and features to predict

        id_column : string
            name of the id column in the metadata

        feature_columns : list
            names of target feature columns

        add_features_columns : list
            names of additional feature columns to add as an input when training

        permutations : list
            list of data permutation functions

        do_permutations : boolen
            either to perfrom data permutations or not

        normalization : function
            normalization function to apply to data

        strategy : object
            tf.distribute object
            used to properly create experimental_distribute_dataset

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

    for path in filepaths:

        data = loadNumpy(path)
        data_list.append(data)

        for feature_idx, feature_column in feature_columns:

            label = getFeaturesFromPath(path, meta, id_column, feature_column)

            if type(label) == int or type(label) == float:
                
                label = np.array(label, dtype=np.int32)

            else:

                label = evaluateString(label)

            label_tensor = tf.convert_to_tensor(label, dtype=tf.float32)
            labels_list[feature_idx].append(label_tensor)

        if add_features_columns is not None:

            for feature_idx, add_feature_columns in enumerate(add_features_columns):

                add_feature = getFeaturesFromPath(path, meta, id_column, add_feature_columns)

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
