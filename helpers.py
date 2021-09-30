import tensorflow as tf
# import module for reading and updating configuration files.
from object_detection.utils import config_util
# import module for building the detection model
from object_detection.builders import model_builder

import numpy as np
import matplotlib.pyplot as plt
import png
import os
import shutil
import gzip
from PIL import Image
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split


def loadJPGToNumpy(filename, dir, image_type):

    """
    Load an image from file into a numpy array.

    parameters
    ----------
    path: string
        a file path

    returns
    -------
    image =/=
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    
    image = Image.open(dir + filename)
    image = np.asarray(image).astype(image_type)
    
    return image


def save2NP(x, output_path):
    """
    saves numpy array to specified path

    parameters
    ----------
    x : ndarray
        any numpy array

    output_path : str
        path where to save numpy array
    """

    try:
        np.save(output_path, x)

    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")


def save2PNG(x, output_path):
    """
    saves numpy array as PNG image to specified path

    parameters
    ----------
    x : ndarray
        numpy array that represents an image, such as spectogram or dicom

    output_path : str
        path where to save PNG image
    """

    try:
        shape = x.shape

        # Convert to float to avoid overflow or underflow losses.
        image_2d = x.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)

        # Write the PNG file
        with open(output_path, 'wb') as png_file:
            w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(png_file, image_2d_scaled)

    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")


def visualizeImage_Box(image, boxes):
    """
    plots an image and (optional) bounding box

    parameters
    ----------
    image : ndarray
        numpy array that represents an image

    boxes:  ndarray
        numpy array that can contain multiple numpy arrays, which represent bounding boxes
        each box should be in the following format: [xmin, ymin, xmax, ymax, ...]
    """

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    if boxes is not None:
        for box in boxes:
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin,
                                   ymax - ymin, fill=False, edgecolor='red'))
        print(boxes)

    print(image.shape)
    # matplotlib inline in notebook
    plt.show()
    # plt.savefig("mygraph.png")


def getPathsList(dir):
    '''
    returns a list of paths to the image files

    parameters
    ----------
    filename: string
        name of the image

    returns: images_paths =/=
        list of paths to each image in directory
    '''

    images_files_list = os.listdir(dir)
    images_paths = [os.path.join(dir, fname) for fname in images_files_list]

    return images_paths


def getLabelFromFilename(filename):

    label = filename.split('_')[-1].replace('.npy', '')
    label = label.split('-')

    return label


def splitTrainValidation(data_folder, train_data_folder, validation_data_folder, counter):

    files_to_split = os.listdir(data_folder)

    files_train, files_val = train_test_split(files_to_split, test_size=0.2)

    for file in files_train:
        shutil.copy(data_folder + file, train_data_folder)
        counter += 1
        print('FINISHED' + str(counter) + '/' + str(len(files_to_split)))
    for file in files_val:
        shutil.copy(data_folder + file, validation_data_folder)
        counter += 1
        print('FINISHED' + str(counter) + '/' + str(len(files_to_split)))


def loadFashionMNIST(dir, reshape_size):

    """
    Loads the Fashion-MNIST gzip dataset.
    Modified by Henry Huang in 2020/12/24.
    We assume that the input_path should in a correct path address format.
    We also assume that potential users put all the four files in the path.

    parameters
    -----------
    dir: string
        path to directory that contains 4 files:
            'train_labels.gz', 'train_images.gz',
            'test_labels.gz', 'test_images.gz'

    returns
    -------
    tuple:
        x_train: ndarray
            contains 28x28 training images
        y_train: ndarray
            contains training class labels as integers
    tuple:
        x_test: ndarray
            contains 28x28 test images
        y_test: ndarray
            contains test class labels as integers
    """

    files = [
        'train_labels.gz', 'train_images.gz',
        'test_labels.gz', 'test_images.gz'
    ]

    paths = []
    for filename in files:
        paths.append(os.path.join(dir, filename))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), reshape_size[0], reshape_size[1])

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), reshape_size[0], reshape_size[1])

    return (x_train, y_train), (x_test, y_test)


def buildClassificationPretrainedModel(model_path, custom_objects, num_classes, activation):

    # custom_objects example: custom_objects={'f1': f1, categorical_focal_loss_fixed': categorical_focal_loss(alpha=[[.25, .25]], gamma=2)}
    loaded_model = tf.keras.models.load_model(model_path, custom_objects)

    model = tf.keras.layers.Sequential()
    # go through all layers until last layer
    for layer in loaded_model.layers[:-1]:
        model.add(layer)

    # classifier
    model.add(tf.keras.layers.Dense(num_classes, activation=activation))

    return model


def buildClassificationImageNetModel(model_imagenet, input_shape, pooling, num_classes, activation):

    loaded_model = model_imagenet(
        include_top=False, weights='imagenet', pooling=pooling, input_shape=input_shape)
    model = tf.keras.models.Sequential()
    model.add(loaded_model)

    # classifier
    model.add(tf.keras.layers.Dense(num_classes, activation=activation))

    return model


def buildDetectionModel(num_classes, checkpoint_path, config_path, dummy_shape):

    # Download the checkpoint and put it into models/research/object_detection/test_data/
    # wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz -O ./efficientdet_d4_1024x1024.tar.gz
    # tar -xf efficientdet_d4_1024x1024.tar.gz
    # mv efficientdet_d4_coco17_tpu-32/checkpoint models/research/object_detection/test_data/
    # tf.keras.backend.clear_session()

    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(
        config_path, config_override=None)

    # Read in the object stored at the key 'model' of the configs dictionary
    model_config = configs['model']

    # Modify the number of classes from its default
    model_config.ssd.num_classes = num_classes

    # Freeze batch normalization
    model_config.ssd.freeze_batchnorm = True

    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    tmp_box_predictor_checkpoint = tf.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads, _box_prediction_head=detection_model._box_predictor._box_prediction_head)

    tmp_model_checkpoint = tf.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor, _box_predictor=tmp_box_predictor_checkpoint)

    # Define a checkpoint
    checkpoint = tf.train.Checkpoint(model=tmp_model_checkpoint)

    # Restore the checkpoint to the checkpoint path
    checkpoint.restore(checkpoint_path)

    # Run a dummy image through the model so that variables are created
    # For the dummy image, you can declare a tensor of zeros that has a shape that the preprocess() method can accept (i.e. [batch, height, width, channels]).
    # use the detection model's `preprocess()` method and pass a dummy image
    dummy = tf.zeros(shape=dummy_shape)
    tmp_image, tmp_shapes = detection_model.preprocess(dummy)

    # run a prediction with the preprocessed image and shapes
    tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

    # postprocess the predictions into final detections
    tmp_detections = detection_model.postprocess(
        tmp_prediction_dict, tmp_shapes)

    tf.keras.backend.set_learning_phase(True)

    return detection_model
