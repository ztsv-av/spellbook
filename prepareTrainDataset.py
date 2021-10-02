import tensorflow as tf

import numpy as np

from helpers import getPathsList, getLabelFromFilename
from permutationFunctions import classification_permutations, detection_permutations


def permuteImageGetLabel(image, permutations, normalization, bboxes, bbox_format, is_detection, is_val):
    
    '''
    permute the given image and get label for that image
    '''

    if is_detection:
        image, bboxes = detection_permutations(image, bboxes, bbox_format, permutations)

    else:    
        if not is_val:
            image = classification_permutations(image, permutations)
    
    image = normalization(image)

    return (image, bboxes) if is_detection else image


def prepareClassificationDataset(batch_size, train_files_path, val_files_path, permutations, normalization, buffer_size, strategy):

    train_paths_list = getPathsList(train_files_path)
    train_len = len(train_paths_list)

    val_paths_list = getPathsList(val_files_path)
    val_len = len(val_paths_list)

    train_images_list = []
    train_labels_list = []
    for path in train_paths_list:
        train_images_list.append(np.load(path))
        train_labels_list.append(getLabelFromFilename(path))
    
    val_images_list = []
    val_labels_list = []
    for path in val_paths_list:
        val_images_list.append(np.load(path))
        val_labels_list.append(getLabelFromFilename(path))

    train_images_map = map(lambda image: permuteImageGetLabel(
        image, permutations, normalization, bboxes=False, bbox_format=False, is_detection=False, is_val=False), train_images_list)
    train_images_mapped_list = list(train_images_map)
    train_dataset_tf = tf.data.Dataset.from_tensor_slices((train_images_mapped_list, train_labels_list))
    train_dataset_tf = train_dataset_tf.shuffle(buffer_size).batch(batch_size)
    train_dataset_dist = strategy.experimental_distribute_dataset(train_dataset_tf)

    val_images_map = val_images_list.map(lambda image: permuteImageGetLabel(
        image, None, normalization, bboxes=False, bbox_format=False, is_detection=False, is_val=True))
    val_images_mapped_list = list(val_images_map)
    val_dataset_tf = tf.data.Dataset.from_tensor_slices((val_images_mapped_list, val_labels_list))
    val_dataset_tf = val_dataset_tf.batch(batch_size)
    val_dataset_dist = strategy.experimental_distribute_dataset(val_dataset_tf)

    return train_dataset_dist, val_dataset_dist, train_len, val_len


def prepareDetectionDataset(filepaths, bbox_format, meta, num_classes, label_id_offset, permutations):

    images_batch = []
    bboxes_batch = []

    for filepath in filepaths:

        filename = filepath.split('/')[-1]
        record = meta[meta['filename'] == filename]
        bboxes = record['bboxes']

        image, bboxes = permuteImageGetLabel(filepath, permutations, bboxes, bbox_format, is_detection=True, is_val=False)

        images_batch.append(image)
        bboxes_batch.append(bboxes)
    
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