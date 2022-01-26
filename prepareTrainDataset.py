from pandas import concat
from helpers import evaluateString, getLabelFromPath, getFeaturesFromPath, loadNumpy, saveNumpyArray
from permutationFunctions import classification_permutations, detection_permutations

import numpy as np
import time
import tensorflow as tf


def permuteImageGetLabelBoxes(
    image, 
    permutations, do_permutations, normalization, 
    is_val, 
    bboxes, bbox_format, is_detection):
    '''
    permutes given image and gets label for that image

    parameters
    ----------
        image : XXX
            XXX

        permutations : XXX
            XXX

        normalization : XXX
            XXX

        bboxes : XXX
            XXX

        bbox_format : XXX
            XXX

        is_detection : XXX
            XXX

        is_val : XXX
            XXX

    returns
    -------
        (image, bboxes) : tuple
            XXX
    '''

    if is_detection:

        image, bboxes = detection_permutations(
            image, bboxes, bbox_format, permutations)

        return (image, bboxes)

    else:
        if not is_val:
            if do_permutations:
                image = classification_permutations(image, permutations)

        image = normalization(image)

        image_tensor = tf.convert_to_tensor(image)

        return image_tensor


def prepareClassificationDataset(
    batch_size, filepaths, 
    meta, id_column, feature_columns, add_features_columns, full_record,
    permutations, do_permutations, normalization, 
    strategy, is_val):
    """
    XXX

    parameters
    ----------

        train_filepaths_batch : XXX
            XXX

        batch_size : XXX
            XXX

        buffer_size : XXX
            XXX

        strategy : XXX
            XXX

        permutations : XXX
            XXX

        normalization : XXX
            XXX

    returns
    -------

        train_dataset_dist_batch : XXX
            XXX
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

            label = getFeaturesFromPath(path, meta, id_column, feature_column, full_record)

            if type(label) == int or type(label) == float:
                
                label = np.array(label, dtype=np.int32)

            else:

                label = evaluateString(label)

            label_tensor = tf.convert_to_tensor(label, dtype=tf.float32)
            labels_list[feature_idx].append(label_tensor)

        if add_features_columns is not None:

            for feature_idx, add_feature_columns in enumerate(add_features_columns):

                add_feature = getFeaturesFromPath(path, meta, id_column, add_feature_columns, full_record)

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

    data_map = map(lambda data: permuteImageGetLabelBoxes(
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
    XXX

    parameters
    ----------

        filepaths : XXX
            XXX

        bbox_format : XXX
            XXX

        meta : XXX
            XXX

        num_classes : XXX
            XXX

        label_id_offset : XXX
            XXX

        permutations : XXX
            XXX

        normalization : XXX
            XXX

    returns
    -------

        images_tensors : XXX
            XXX

        bboxes_tensors : XXX
            XXX

        classes_one_hot_tensors : XXX
            XXX
    """

    images_batch = []
    bboxes_batch = []

    for filepath in filepaths:

        filename = filepath.split('/')[-1]
        record = meta[meta['filename'] == filename]

        image = np.load(filepath)
        bboxes = evaluateString(record['bboxes'].values[0])

        image, bboxes = permuteImageGetLabelBoxes(
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
