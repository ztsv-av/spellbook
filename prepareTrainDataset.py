import tensorflow as tf

import numpy as np

from helpers import getPathsList, getLabelFromFilename
from permutationFunctions import classification_permutations, detection_permutations


def permuteImageGetLabel(image_filename, image_dir, permutations, bboxes, bbox_format, is_detection, is_val):
    
    '''
    permute the given image and get label for that image
    '''

    image = np.load(image_filename + image_dir)
    
    if is_detection:
        image, bboxes = detection_permutations(image, bboxes, bbox_format, permutations)

    else:    
        if not is_val:
            image = classification_permutations(image, permutations)
        label = getLabelFromFilename(image_filename)

    return (image, bboxes) if is_detection else (image, label)


def prepareClassificationDataset(batch_size, train_files_path, val_files_path, permutations, buffer_size):

    train_paths_list = getPathsList(train_files_path)
    train_len = len(train_paths_list)

    val_paths_list = getPathsList(val_files_path)
    val_len = len(val_paths_list)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths_list)
    train_dataset = train_dataset.map(lambda x: permuteImageGetLabel(
        x, train_files_path, permutations, bboxes=False, bbox_format=False, is_detection=False, is_val=False))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_paths_list)
    val_dataset = val_dataset.map(lambda x: permuteImageGetLabel(
        x, val_files_path, permutations=None, bboxes=False, bbox_format=False, is_detection=False, is_val=True))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset, train_len, val_len


def prepareDetectionDataset(filenames, files_path, bbox_format, meta, num_classes, label_id_offset, permutations):

    images_batch = []
    bboxes_batch = []

    for filename in filenames:

        record = meta[meta['filename'] == filename]
        bboxes = record['bboxes']

        image, bboxes = permuteImageGetLabel(filename, files_path, permutations, bboxes, bbox_format, is_detection=True, is_val=False)

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