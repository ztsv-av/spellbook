import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import librosa
import albumentations
import numpy as np
import random

import tensorflow as tf
import tensorflow.keras.backend as K
from torchvision import transforms


def minMaxNormalizeNumpy(x):
    """
    normalizes image input to [0, 1] interval

    parameters
    ----------
        x : ndarray
            input image

    returns
    -------
        x : ndarray
            normalized image
    """

    if x.dtype != 'float32':
        x = np.float32(x)

    try:
        x -= x.min()
        x /= x.max()

        return x

    except AttributeError:
        raise TypeError("first convert x to numpy array")


def minMaxNormalizeTensor(x):
    """
    normalizes tensor image to [0, 1] interval

    parameters
    ----------
        x : tensor
            input image

    returns
    -------
        x : ndarray
            normalized tensor image
    """

    x = tf.subtract(x, tf.reduce_min(x))
    x = tf.divide(x, tf.reduce_max(x))

    return x


def meanStdNormalize(x):
    """
    normalizes image input to [0, 1] interval

    parameters
    ----------
        x : ndarray
            input image

    returns
    -------
        x : ndarray
            normalized image
    """

    try:
        x -= np.mean(x)
        x /= np.std(x)

        return x

    except AttributeError:
        raise TypeError("first convert x to numpy array")


def kerasNormalize(model_name):
    """
    initializes a normalization function based on the ImageNet model

    parameters
    ----------
        model_name : string
            name of the ImageNet model

    returns
    -------
        normalization_function : function
            tf.keras.applications preprocess function
    """


    if model_name == 'VGG16':

        normalization_function = tf.keras.applications.vgg16.preprocess_input

    elif model_name == 'VGG19':

        normalization_function = tf.keras.applications.vgg19.preprocess_input
    
    elif model_name == 'InceptionV3':

        normalization_function = tf.keras.applications.inception_v3.preprocess_input
        
    elif model_name == 'Xception':

        normalization_function = tf.keras.applications.xception.preprocess_input
        
    elif model_name == 'MobileNet':

        normalization_function = tf.keras.applications.mobilenet.preprocess_input

    elif model_name == 'MobileNetV2':

        normalization_function = tf.keras.applications.mobilenet_v2.preprocess_input
    
    elif model_name == 'InceptionResNetV2':

        normalization_function = tf.keras.applications.inception_resnet_v2.preprocess_input
    
    elif (model_name == 'ResNet50') or (model_name == 'ResNet101'):

        normalization_function = tf.keras.applications.resnet50.preprocess_input
        
    elif (model_name == 'ResNet50V2') or (model_name == 'ResNet101V2'):

        normalization_function = tf.keras.applications.resnet_v2.preprocess_input

    elif 'DenseNet' in model_name:

        normalization_function = tf.keras.applications.densenet.preprocess_input

    elif 'EfficientNet' in model_name:

        normalization_function = minMaxNormalizeTensor

    return normalization_function


def addColorChannels(x, num_channels):
    """
    adds channel dimension to 2D image

    parameters
    ----------
        x : ndarray
            input image

        numChannels : int
            number of channels to add (to copy) into channel dimension

    returns
    -------
        x : ndarray
            image with channel dimension
    """

    try:
        x = np.repeat(x[..., np.newaxis], num_channels, -1)

        return x

    except AttributeError:
        raise TypeError("first convert x to numpy array")


def spectrogramToDecibels(x):
    """
    converts a power spectrogram (amplitude squared) to decibel (dB) units

    parameters
    ----------
        x : ndarray
            input power spectogram

    returns
    -------
        x : ndarray
            decibel spectogram
    """

    try:
        x = librosa.power_to_db(x.astype(np.float32), ref=np.max)

        return x

    except AttributeError:
        raise TypeError("first convert x to numpy array")


def normalizeSpectogram(x):
    """
    normalizes spectogram

    parameters
    ----------
        x : ndarray
            input spectogram

    returns
    -------
        x : ndarray
            normalized spectogram
    """

    try:
        x = (x + 80) / 80

        return x

    except AttributeError:
        raise TypeError("first convert x to numpy array")


def normalizeBBox(xmin, ymin, xmax, ymax, image_shape):
    """
    normalizes box coordinates  to [0, 1] interval

    parameters
    ----------

        xmin : float
            box left coordinate on x-axis

        ymin : float
            box bottom coordinate on y-axis

        xmax : float
            box right coordinate on x-axis

        ymax : float
            box top coordinate on y-axis

    returns
    -------
        box_n : list
            bounding box with normalized coordinates in format [xmin, ymin, xmax, ymax]
    """
    xmin_n = xmin / image_shape[1]
    ymin_n = ymin / image_shape[0]
    xmax_n = xmax / image_shape[1]
    ymax_n = ymax / image_shape[0]

    box_n = [xmin_n, ymin_n, xmax_n, ymax_n]

    return box_n


def dicomToArray(dicom_path, voi_lut=True, fix_monochrome=True):
    """
    converts dicom file to numpy array

    parameters
    ----------
        dicom_path : str
            full path to dicom file (for example, '.../dicom_dir/filename.dcm')

        voi_lut : boolean, default is True
            when True, transforms DICOM data to more simpler data

        fix_monochrome : boolean, default is True
            when True, fixes X-ray that looks inverted

    returns
    -------
        image_numpy : ndarray
            converted dicom image to numpy array
    """

    dcm_file = pydicom.read_file(dicom_path)

    if voi_lut:
        image_numpy = apply_voi_lut(dcm_file.pixel_array, dcm_file)
    else:
        image_numpy = dcm_file.pixel_array

    if fix_monochrome and dcm_file.PhotometricInterpretation == "MONOCHROME1":
        image_numpy = np.amax(image_numpy) - image_numpy

    image_numpy = np.float32(image_numpy)

    return image_numpy


def ratioResize(image, boxes):
    """
    crops image to size where height = width and changes values (location) of pixels in bounding boxes accordingly

    parameters
    ----------
        image : ndarray
            numpy array that represents image

        boxes : list
            either list of coordinates of bounding box, or list of lists of coordinates of bounding boxes
            coordinates must follow format [xmin, ymin, xmax, ymax, 'class_id']

    returns
    -------
        image : ndarray
            cropped image

        boxes : list
            consistent representation of bounding boxes for cropped images
    """

    height = image.shape[0]
    width = image.shape[1]

    if height < width:
        diff2 = (width - height) // 2
        residual = (width - 2 * diff2) - height
        image = image[:, diff2: width - diff2 - residual]

        if boxes:

            for idx, box in enumerate(boxes):
                xmin, xmax = box[0], box[2]

                xmin -= diff2
                xmax -= diff2

                if xmin < 0:
                    xmin = 0

                if xmin > image.shape[1]:
                    del boxes[idx]
                    continue

                if xmax > image.shape[1]:
                    xmax = image.shape[1]

                box[0], box[2] = xmin, xmax
                boxes[idx] = box

    elif width < height:
        diff2 = (height - width) // 2
        residual = (height - 2 * diff2) - width
        image = image[diff2: height - diff2 - residual, :]

        if boxes:

            for idx, box in enumerate(boxes):
                ymin, ymax = box[1], box[3]

                ymin -= diff2
                ymax -= diff2

                if ymin < 0:
                    ymin = 0

                if ymax > image.shape[0]:
                    ymax = image.shape[0]

                box[1], box[3] = ymin, ymax
                boxes[idx] = box

    return image, boxes


def resizeImageBbox(image, bboxes, height, width, bbox_format):
    """
    resizes image to shape given shape (height, width) and changes values (location) of pixels in bounding boxes accordingly
    uses albumentation (source https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.Resize)

    parameters
    ----------
        image : ndarray
            numpy array that represents image

        boxes : list
            either list of coordinates of bounding box, or list of lists of coordinates of bounding boxes
            coordinates must follow format [xmin, ymin, xmax, ymax, label]

        height : int
            desired height of image

        width : int
            desired width of an image

    returns
    -------
        image : ndarray
            cropped image

        boxes : list
            consistent representation of bounding boxes for cropped images
    """

    # create resize transform pipeline
    transform = albumentations.Compose(
        [albumentations.Resize(height=height, width=width, always_apply=True)],
        bbox_params=albumentations.BboxParams(format=bbox_format))

    transformed = transform(image=image, bboxes=bboxes)
    image = transformed['image']
    bboxes = transformed['bboxes']

    return image, bboxes


def randomMelspecPower(data, power, c):

    data -= data.min()
    data /= (data.max() + K.epsilon())
    data **= (random.random() * power + c)
    
    return data    


def melspecMonoToColor(x:np.ndarray, input_shape, normalization):

    x = addColorChannels(x, input_shape[-1])
    v = (255 * x)
    if normalization is not None:
        v = normalization(v)

    return v
