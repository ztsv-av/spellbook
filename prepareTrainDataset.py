from helpers import evaluateString, getLabelFromFilename, loadNumpy
from permutationFunctions import classification_permutations, detection_permutations

import numpy as np
import time
import tensorflow as tf


def permuteImageGetLabelBoxes(image, permutations, do_permutations, normalization, is_val, bboxes, bbox_format, is_detection):
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
        if not is_val and do_permutations:
            image = classification_permutations(image, permutations)

        image = normalization(image)

        image_tensor = tf.convert_to_tensor(image)

        return image_tensor


def prepareClassificationDataset(batch_size, filepaths_part, permutations, do_permutations, normalization, strategy, is_val):
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

    # start_load_numpy_time = time.time()
    # print('Start Loading Numpy Files...', flush=True)

    images_list_part = []
    labels_list_part = []
    for path in filepaths_part:

        image = loadNumpy(path)
        images_list_part.append(image)

        label = getLabelFromFilename(path)
        label_tensor = tf.convert_to_tensor(label)
        labels_list_part.append(label_tensor)

    # end_load_numpy_time = time.time()
    # print('Finished loading Numpy Files. Time Passed: ' + str(end_load_numpy_time - start_load_numpy_time), flush=True)

    # start_mapping_data_time = time.time()
    # print('Start Mapping Data...', flush=True)

    images_map = map(lambda image: permuteImageGetLabelBoxes(
        image, permutations, do_permutations, normalization, is_val, bboxes=None, bbox_format=None, is_detection=False), images_list_part)
    images_mapped_list_part = list(images_map)

    # end_mapping_data_time = time.time()
    # print('Finished Mapping Data. Time Passed: ' + str(end_mapping_data_time - start_mapping_data_time), flush=True)

    # start_creating_dataset_time = time.time()
    # print('Start Creating Tensorflow Dataset...', flush=True)

    data_part = tf.data.Dataset.from_tensor_slices(
        (images_mapped_list_part, labels_list_part))

    # end_creating_dataset_time = time.time()
    # print('Finished Creating Tensorflow Dataset. Time Passed: ' + str(end_creating_dataset_time - start_creating_dataset_time), flush=True)

    data_part = data_part.batch(batch_size)
    data_part_dist = strategy.experimental_distribute_dataset(
        data_part)

    return data_part_dist


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
