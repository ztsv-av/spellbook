import cv2
import numpy as np

from helpers import visualizeImageBox
from preprocessFunctions import dicomToArray, ratioResize, minMaxNormalizeNumpy, addColorChannels


def resizeImage(image, h, w):

    # create resize transform pipeline
    transformed = cv2.resize(image, dsize=(h, w), interpolation=cv2.INTER_LANCZOS4)

    return transformed


def testPreprocessing(dcm_name, dcm_path, classification):

    # convert dcm to numpy
    image_inital = dicomToArray(dcm_name, dcm_path)
    # make image as square
    # image = ratio_convert(image_inital)
    # resize image
    if classification:
        image = resizeImage(image_inital, 256, 256)
        image = minMaxNormalizeNumpy(image)
    else:
        image = resizeImage(image_inital, 512, 512)
        image = (np.maximum(image,0) / image.max()) * 255.

    # add 3 channels
    image = addColorChannels(image, 3)

    return image, image_inital.shape


def labelDecoder(idx):

    label = ''

    if idx == 0:
        label = 'negative'
    elif idx == 1:
        label = 'typical'
    elif idx == 2:
        label = 'indeterminate'
    else:
        label = 'atypical'

    return label
