import os
import gzip
import png
import ast
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.model_selection import train_test_split

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


def loadImage(path, image_type):
    """
    loads input image as numpy array

    parameters
    ----------
        path: string
            full path to input image

        image_type : string or dtype
            Typecode or data-type to which the array is cast.

    returns
    -------
        image_converted : ndarray
            numpy array representing image of shape (height, width, 3)
    """

    image = Image.open(path)
    image_converted = np.asarray(image).astype(image_type)

    return image_converted


def saveNumpyArray(x, path):
    """
    saves numpy array to specified directory

    parameters
    ----------
        x : ndarray
            numpy array

        path : str
            full path where to save x
    """

    try:
        np.save(path, x)

    except AttributeError:
        raise TypeError("input image is not numpy array")


def saveImage(image, path):
    """
    saves loaded jpg image to specified directory

    parameters
    ----------
        image : ndarray
            numpy array

        path : str
            full path where to save an image
    """

    image = Image.fromarray(image)
    image.save(path)


def convertNumpy2png(x, path):
    """
    saves numpy array as .png image to specified directory

    parameters
    ----------
        x : ndarray
            numpy array representing image (such as spectogram or dicom)

        path : str
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
        with open(path, 'wb') as png_file:
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


def visualizeImageBox(image, boxes):
    """
    plots input image and bounding box (optional)

    parameters
    ----------
        image : ndarray
            numpy array representing image

        boxes : list (list of lists)
            list of coordinates of bounding box (or list of multiple bounding boxes)
            each box must follow format [ymin, xmin, ymax, xmax, ...]
    """

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    if boxes is not None:
        for box in boxes[:3]:
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
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
            (each holding category index `id` and category name `name`) keyed with category indices

        score_threshold : float
            minimum score threshold for a box or keypoint to be visualized
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


def getLabelFromPath(path):
    """
    returns list of labels corresponding to the file path 
    filename is itself a sequence of labels separated by the dash symbol

    parameters
    ----------
        path : string
            full path to the image file

    returns
    -------
        label : list of integers
            labels corresponding to the file
    """

    label = path.split('_')[-1].replace('.npy', '')
    label = label.split('-')
    label = list(map(int, label))

    return label


def getFeaturesFromPath(path, meta, id_column, feature_column, filename_underscore):
    """
    returns row or a cell data from the dataframe

    parameters
    ----------
        path : string
            full path to the image file
        
        meta : dataframe
            contains columns with data

        id_column : string
            name of the filename id column in the dataframe
        
        feature_column : string
            name of the column from which to load the data

        filename_underscore : boolean
            True if end of a filename has a class after an underscore

    returns
    -------
        features : string or a number
            data from the desired column of the dataframe
    """

    file_id = path.split('/')[-1]

    if filename_underscore:

        file_id = file_id.split('_')[0]

    record = meta[meta[id_column] == file_id]

    features = record[feature_column].values[0]

    return features


def findNLargest(array, n):

    sorted_array = sorted(array)
    
    n_largest = []
    check = 0
    count = 1
 
    for i in range(1, len(sorted_array) + 1):
 
        if(count < n + 1):

            if(check != sorted_array[n - i]):
                 
                # handles duplicate values 
                n_largest.append(sorted_array[n - i])
                check = sorted_array[n - i]
                count += 1

        else:

            break
    
    return n_largest


def splitTrainValidation(dir, train_data_folder, val_data_folder, val_data_size=0.2):
    """
    reads files in input directory, splits and copies them into two directories
    using specified ratio

    parameters
    ----------
        dir : string
            path to directory with files to split

        train_data_folder : string
            path to directory where to store train files

        val_data_folder : string
            path to directory where to store validation files

        val_data_size : float, default is 0.2
            how much percent of data to allocate for validation
    """

    files_to_split = os.listdir(dir)
    files_train, files_val = train_test_split(
        files_to_split, test_size=val_data_size)

    if not os.path.exists(train_data_folder):
        os.makedirs(train_data_folder)

    if not os.path.exists(val_data_folder):
        os.makedirs(val_data_folder)

    for file in files_train:
        shutil.copy(dir + file, train_data_folder)

    for file in files_val:
        shutil.copy(dir + file, val_data_folder)


def loadFashionMNIST(dir, reshape_size):
    """
    loads the Fashion-MNIST gzip dataset

    parameters
    -----------
        dir : string
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


def deleteFiles(path, to_delete):
    """
    randomly deletes a number of files from a given directory

    parameters
    -----------
        path : string
            path to directory where to delete files

        to_delete : int
            number of files to delete
            cannot exceed total number of files in a directory
    """

    files = os.listdir(path)
    random_idxs = random.sample(range(0, len(files)), to_delete)

    for random_idx in random_idxs:

        file_to_delete = files[random_idx]
        os.remove(path + file_to_delete)


def createOneHotVector(path, right_class_idx, num_classes):
    """
    creates a one-hot list with 1 at class index

    parameters
    -----------
        path : string
            full path to a file

        right_class_idx : int
            which idx to use when splitting path my underscore

        num_classes : int
            number of classes, lenght of the one-hot array
    """

    class_idx = evaluateString(path.split('_')[-right_class_idx].replace('.npy', ''))

    onehot_numpy = np.zeros(num_classes)
    onehot_numpy[class_idx] = 1
    onehot_list = onehot_numpy.astype(int).tolist()

    return onehot_list


def createSparseValue(path, right_class_idx):
    """
    creates a one-hot list with 1 at class index

    parameters
    -----------
        path : string
            full path to a file

        right_class_idx : int
            which idx to use when splitting path my underscore
    """

    class_idx = evaluateString(path.split('_')[-right_class_idx].replace('.npy', ''))
    class_idx = int(class_idx)

    return class_idx
