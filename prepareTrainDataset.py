from helpers import evaluateString, getLabelFromPath, getFeaturesFromPath, loadNumpy, saveNumpyArray
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
        if not is_val:
            if do_permutations:
                image = classification_permutations(image, permutations)

        image = normalization(image)

        image_tensor = tf.convert_to_tensor(image)

        return image_tensor


def prepareClassificationDataset(
    batch_size, filepaths, meta, id_column, feature_column, full_record,
    permutations, do_permutations, normalization, strategy, is_autoencoder, is_val):
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

    if is_autoencoder:

        data_list = []
        features_list = []

        for path in filepaths:

            data = loadNumpy(path)
            data_list.append(data)

            features = getFeaturesFromPath(path, meta)
            features_tensor = tf.cast(tf.convert_to_tensor(features), dtype=tf.float32)
            features_list.append(features_tensor)

        data_dataset = tf.data.Dataset.from_tensor_slices(
            (data_list, features_list))

        data_dataset = data_dataset.batch(batch_size)

        data_dataset_dist = strategy.experimental_distribute_dataset(
            data_dataset)

        return data_dataset_dist


    else:

        data_list = []

        # labels_list = []

        labels_t_list = []
        labels_b_list = []
        labels_f_list = []
        labels_s_list = []
        labels_c_list = []
        labels_a_list = []

        for path in filepaths:

            data = loadNumpy(path)
            data_list.append(data)

            # label = getFeaturesFromPath(path, meta, id_column, feature_column, full_record)

            label_t = getFeaturesFromPath(path, meta, id_column, 'type', full_record)
            label_b = getFeaturesFromPath(path, meta, id_column, 'breed', full_record)
            label_f = getFeaturesFromPath(path, meta, id_column, 'fur', full_record)
            label_s = getFeaturesFromPath(path, meta, id_column, 'size', full_record)
            label_c = getFeaturesFromPath(path, meta, id_column, 'color', full_record)
            label_a = getFeaturesFromPath(path, meta, id_column, 'age', full_record)

            # if feature_column != 'popularity':
                
            #     label = evaluateString(label)

            # else:

            #     label = [label]
            #     label = np.array(label, dtype=np.int32)
                
            label_t = evaluateString(label_t)
            label_b = evaluateString(label_b)
            label_f = evaluateString(label_f)
            label_s = evaluateString(label_s)
            label_c = evaluateString(label_c)
            label_a = evaluateString(label_a)

            # label_tensor = tf.convert_to_tensor(label, dtype=tf.float32)
            # labels_list.append(label_tensor)

            label_t_tensor = tf.convert_to_tensor(label_t, dtype=tf.float32)
            labels_t_list.append(label_t_tensor)
            label_b_tensor = tf.convert_to_tensor(label_b, dtype=tf.float32)
            labels_b_list.append(label_b_tensor)
            label_f_tensor = tf.convert_to_tensor(label_f, dtype=tf.float32)
            labels_f_list.append(label_f_tensor)
            label_s_tensor = tf.convert_to_tensor(label_s, dtype=tf.float32)
            labels_s_list.append(label_s_tensor)
            label_c_tensor = tf.convert_to_tensor(label_c, dtype=tf.float32)
            labels_c_list.append(label_c_tensor)
            label_a_tensor = tf.convert_to_tensor(label_a, dtype=tf.float32)
            labels_a_list.append(label_a_tensor)

        data_map = map(lambda data: permuteImageGetLabelBoxes(
            data, permutations, do_permutations, normalization, is_val, bboxes=None, bbox_format=None, is_detection=False), data_list)
        data_mapped_list = list(data_map)

        data_dataset = tf.data.Dataset.from_tensor_slices(
            ((data_mapped_list, labels_t_list, labels_b_list, labels_f_list, labels_s_list, labels_c_list, labels_a_list), (labels_t_list, labels_b_list, labels_f_list, labels_s_list, labels_c_list, labels_a_list)))

        data_dataset = data_dataset.batch(batch_size)

        data_dataset_dist = strategy.experimental_distribute_dataset(
            data_dataset)

        return data_dataset_dist

        # data_list = []
        # labels_list = []

        # for path in filepaths:

        #     data = loadNumpy(path)
        #     data_list.append(data)

        #     label = getLabelFromPath(path)
        #     label_tensor = tf.convert_to_tensor(label)
        #     labels_list.append(label_tensor)

        # data_map = map(lambda data: permuteImageGetLabelBoxes(
        #     data, permutations, do_permutations, normalization, is_val, bboxes=None, bbox_format=None, is_detection=False), data_list)
        # data_mapped_list = list(data_map)

        # data_dataset = tf.data.Dataset.from_tensor_slices(
        #     (data_mapped_list, labels_list))

        # data_dataset = data_dataset.batch(batch_size)

        # data_dataset_dist = strategy.experimental_distribute_dataset(
        #     data_dataset)

        # return data_dataset_dist

    # start_load_numpy_time = time.time()
    # print('Start Loading Numpy Files...', flush=True)

    # end_load_numpy_time = time.time()
    # print('Finished loading Numpy Files. Time Passed: ' + str(end_load_numpy_time - start_load_numpy_time), flush=True)

    # start_mapping_data_time = time.time()
    # print('Start Mapping Data...', flush=True)

    # end_mapping_data_time = time.time()
    # print('Finished Mapping Data. Time Passed: ' + str(end_mapping_data_time - start_mapping_data_time), flush=True)

    # start_creating_dataset_time = time.time()
    # print('Start Creating Tensorflow Dataset...', flush=True)

    # end_creating_dataset_time = time.time()
    # print('Finished Creating Tensorflow Dataset. Time Passed: ' + str(end_creating_dataset_time - start_creating_dataset_time), flush=True)



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
