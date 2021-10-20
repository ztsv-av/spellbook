import os
import png
import gzip
import ast
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils


def loadNumpy(path):
    """
    loads numpy file from specified directory

    parameters
    ----------
    path : string
        full path to file

    returns
    -------
    numpy_file : ndarray
        numpy array
    """

    numpy_file = np.load(path)

    return numpy_file


def saveImage2Numpy(filename, dir, image_type):
    """
    saves input image as numpy array to specified directory

    parameters
    ----------
    filename: string
        name of input image

    dir : string
        full path to input image (up to folder)

    image_type : XXX
        XXX

    returns
    -------
    image : ndarray
        array representing image of shape (height, width, 3)
    """

    image = Image.open(dir + filename)
    image_converted = np.asarray(image).astype(image_type)

    return image_converted


def saveNumpyArray(x, dir):
    """
    saves numpy array to specified directory

    parameters
    ----------
    x : ndarray
        numpy array

    dir : str
        full path where to save x
    """

    try:
        np.save(dir, x)

    except AttributeError:
        raise TypeError("input image is not numpy array")


def convertNumpy2png(x, dir):
    """
    saves numpy array as .png image to specified directory

    parameters
    ----------
    x : ndarray
        numpy array representing image (such as spectogram or dicom)

    dir : str
        full path where to save x as .png image
    """

    try:
        shape = x.shape

        # convert to float to avoid overflow or underflow losses
        image_2d = x.astype(float)

        # rescale grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

        # convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)

        # write the .png file
        with open(dir, 'wb') as png_file:
            w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(png_file, image_2d_scaled)

    except AttributeError:
        raise TypeError("input image is not numpy array")


def evaluateString(x):
    """
    evaluates expression node or string containing Python literal or container display
    provided string or node must consist of strings, bytes, numbers, tuples, lists, dictionaries, booleans and None

    parameters
    ----------
    x : string
        string which needs to be evaluated ('unstringified')

    returns
    -------
    x : string
        evaluated input string
    """

    x = ast.literal_eval(x)

    return x


def visualizeImage_Box(image, boxes):
    """
    plots input image and bounding box (optional)

    parameters
    ----------
    image : ndarray
        numpy array representing image

    boxes : list (list of lists)
        list of coordinates of bounding box (or list of multiple bounding boxes)
        each box must follow format [xmin, ymin, xmax, ymax, ...]
    """

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    if boxes is not None:
        for box in boxes[:3]:
            xmin, ymin, xmax, ymax = box[1], box[0], box[3], box[2]
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin,
                                   ymax - ymin, fill=False, edgecolor='red'))

    plt.show()


def visualizeDetections(image, boxes, classes, scores, category_index, score_threshold):
    """
    wrapper to visualize detections

    parameters
    ----------
    image_np: ndarray
        numpy array of shape (height, width, 3)

    boxes: list
        list of shape [N, 4] where N is number of boxes

    classes: ndarray
        numpy array of shape [M] where M is number of classes
        note that class indices are 1-based and match keys in label map

    scores: ndarray
        numpy array of shape [K] where K is number of scores
        if scores = None then boxes are assumed groundtruth and plotted with black color without class and score labels

    category_index: dict
        dictionary containing category dictionaries
        (each holding category index `id` and category name `name`) keyed with category indices XXX

    score_threshold : float
        #TODO
    """

    image_annotations = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_annotations, boxes, classes, scores, category_index,
        use_normalized_coordinates=True, min_score_thresh=score_threshold)

    plt.imshow(image_annotations)


def getFullPaths(dir):
    '''
    returns list of full paths to files in given directory

    parameters
    ----------
    dir: string
        directory with files

    returns
    -------
    paths : list
        list of paths for each image in dir
    '''

    filenames = os.listdir(dir)
    paths = [os.path.join(dir, fname) for fname in filenames]

    return paths


def getLabelFromFilename(filename):
    """
    XXX

    parameters
    ----------
    filename : XXX
        XXX

    returns
    -------
    label : XXX
        XXX
    """

    label = filename.split('_')[-1].replace('.npy', '')
    label = label.split('-')
    label = list(map(int, label))

    return label


def splitTrainValidation(dir, train_data_folder, val_data_folder, val_data_size=0.2):
    """
    XXX

    parameters
    ----------
    dir : XXX
        XXX

    train_data_folder : XXX
        XXX

    val_data_folder : XXX
        XXX

    val_data_size : float, default is 0.2
        how much percent of data to allocate for validation
    """

    files_to_split = os.listdir(dir)
    files_train, files_val = train_test_split(
        files_to_split, test_size=val_data_size)

    train_counter = 0
    val_counter = 0

    for file in files_train:
        shutil.copy(dir + file, train_data_folder)
        train_counter += 1
        print('FINISHED' + str(train_counter) + '/' + str(len(files_to_split)))

    for file in files_val:
        shutil.copy(dir + file, val_data_folder)
        val_counter += 1
        print('FINISHED' + str(val_counter) + '/' + str(len(files_to_split)))


def loadFashionMNIST(dir, reshape_size):
    """
    loads the Fashion-MNIST gzip dataset

    parameters
    -----------
    dir: string
        path to directory that contains 4 files:
            train_labels.gz, train_images.gz,
            test_labels.gz, test_images.gz

    returns
    -------
    (x_train, y_train) : tuple
        x_train: ndarray
            contains 28x28 training images
        y_train: ndarray
            contains training class labels as integers
    (x_test, y_test) : tuple
        x_test: ndarray
            contains 28x28 test images
        y_test: ndarray
            contains test class labels as integers
    """

    files = [
        'train_labels.gz', 'train_images.gz',
        'test_labels.gz', 'test_images.gz']

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
    """
    XXX

    parameters
    ----------
    model_path : XXX
        XXX

    custom_objects : XXX
        XXX

        example:
        custom_objects = {
            'f1': f1,
            'categorical_focal_loss_fixed': categorical_focal_loss(alpha=[[.25, .25]], gamma=2)}

    num_classes : XXX
        XXX

    activation : XXX
        XXX

    returns
    -------
    model : XXX
        XXX
    """

    loaded_model = tf.keras.models.load_model(model_path, custom_objects)

    model = tf.keras.layers.Sequential()
    # add all layers except last layer
    for layer in loaded_model.layers[:-1]:
        model.add(layer)

    # add last classification layer
    model.add(tf.keras.layers.Dense(num_classes, activation=activation))

    return model


def buildClassificationImageNetModel(model_imagenet, input_shape, pooling, num_classes, activation):
    """
    builds classification ImageNet model given image input shape, number of classes, pooling and activation layers

    parameters
    ----------
    model_imagenet : XXX
        XXX

    input_shape : XXX
        XXX

    pooling : XXX
        XXX

    num_classes : int
        number of classes in dataset

    activation : XXX
        XXX

    returns
    -------
    model : XXX
        XXX
    """

    loaded_model = model_imagenet(
        include_top=False, weights='imagenet', pooling=pooling, input_shape=input_shape)
    model = tf.keras.models.Sequential()
    model.add(loaded_model)

    # add last classification layer
    model.add(tf.keras.layers.Dense(num_classes, activation=activation))

    return model


def buildDetectionModel(num_classes, checkpoint_path, config_path, dummy_shape):
    """
    # TODO : napiwi tyt description, pomesti kommenti v description kakie nado -- ostal'nie ydali
    """

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
