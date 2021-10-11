import tensorflow as tf

import pydicom
import numpy as np
import albumentations
import librosa
from pydicom.pixel_data_handlers.util import apply_voi_lut


def minMaxNormalizeNumpy(x):

    """
    normalizes image input to [0, 1] interval

    parameters
    ----------
    x : ndarray
        input image

    returns
    -------
    x : =/=
        normalized image
    """

    try:
        x -= x.min()
        x /= x.max()

        return x

    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")


def minMaxNormalizeTensor(x):

    """
    normalizes tensor image to [0, 1] interval

    parameters
    ----------
    x : tensor
        input image

    returns
    -------
    x : =/=
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
    x : =/=
        normalized image
    """

    try:
        x -= np.mean(x)
        x /= np.std(x)

        return x

    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")


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
    x : =/=
        image with channel dimension
    """

    try:
        x = np.repeat(x[..., np.newaxis], num_channels, -1)

        return x

    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")


def spectrogramToDecibels(x):

    """
    converts a power spectrogram (amplitude squared) to decibel (dB) units

    parameters
    ----------
    x : ndarray
        input power spectogram

    returns
    -------
    x : =/=
        decibel spectogram
    """

    try:
        x = librosa.power_to_db(x, ref=np.max)

        return x

    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")


def normalizeSpectogram(x):

    """
    normalizes spectogram

    parameters
    ----------
    x : ndarray
        input spectogram

    returns
    -------
    x : =/=
        normalized spectogram
    """

    try:
        x = (x + 80) / 80

        return x

    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")

def normalizeBBox(ymin, xmin, ymax, xmax, image_shape):

    return [ymin / image_shape[1], xmin / image_shape[0], ymax / image_shape[1], xmax / image_shape[0]]


def dicomToArray(dicom_path, voi_lut=True, fix_monochrome=True):

    """
    converts dicom file to numpy array

    parameters
    ----------
    dicom_path : str
        path to dicom file
        format: 'dicom_dir/filename.dcm'

    voi_lut: bool
        used to transform DICOM data to more simpler data

    fix_monochrome: bool
        used to fix X-ray that looks inverted

    returns
    -------
    image_numpy : =/=
        dicom numpy array
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
    cropps image to size where height = width and 
    changes values (location) of pixels in bounding boxes accordingly

    parameters
    ----------
    image : ndarray
        numpy array that represents an image

    boxes : 
        boxes:  ndarray
        numpy array that can contain multiple numpy arrays, which represent bounding boxes
        each box should be in the following format: [xmin, ymin, xmax, ymax, 'class_id']

    returns
    -------
    image : =/=
        cropped image
    boxes : =/=
        bounding boxes with changed pixel locations
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
    resizes image to shape (height, width) where height = width and 
    changes values (location) of pixels in bounding boxes accordingly

    parameters
    ----------
    image : ndarray
        numpy array that represents an image

    boxes : 
        boxes:  ndarray
        numpy array that can contain multiple numpy arrays, which represent bounding boxes
        each box should be in the following format: [xmin, ymin, xmax, ymax, label]

    height : int
        desired height of an image

    width : int
        desired width of an image

    returns
    -------
    transformed : =/=
        dictionary containing {'image': resized image, 'bboxes': boxes of format [xmin, ymin, xmax, ymax, label]}
    """

    # create resize transform pipeline
    transform = albumentations.Compose(
        [albumentations.Resize(height=height, width=width, always_apply=True)],
        bbox_params=albumentations.BboxParams(format=bbox_format))

    transformed = transform(image=image, bboxes=bboxes)
    image = transformed['image']
    bboxes = transformed['bboxes']

    return image, bboxes
