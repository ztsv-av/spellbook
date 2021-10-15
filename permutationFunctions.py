import random
import numpy as np
import albumentations as A


def whiteNoise(x, input_shape, noise_level, p):
    """
    adds white noise to input melspectogram

    parameters
    ----------
    x : ndarray of shape input_shape
        input melspectogram
        taken from globalVariables.py

    inputShape : tuple
        shape of melspectogram
        taken from globalVariables.py

    noiseLevel : float
        noise multiplier. Lower values correspond to lower frequencies, and vice versa
        taken from globalVariabels.py

    p : float
        probability to apply white noise to input melspectogram
        taken from globalVariables.py

    returns
    -------
    x : =/=
        white-noised image
    """

    if random.random() < p:
        white_noise = (np.random.sample((input_shape[0], input_shape[1])).astype(
            np.float32) + 9) * x.mean() * noise_level * (np.random.sample() + 0.3)
        x = x + white_noise

    return x


def bandpassNoise(x, input_shape, noise_level, p):
    """
    selects a random interval of length 20 in the first half of input melspectogram, and adds bandpass noise to this interval

    parameters
    ----------
    x : list or ndarray of shape input_shape
        input melspectogram
        taken from globalVariables.py

    input_shape : tuple
        shape of melspectogram
        taken from globalVariables.py

    noise_level : float
        noise multiplier. Lower values correspond to lower frequencies, and vice versa
        taken from globalVariabels.py

    p : float
        probability to apply white noise to input melspectogram
        taken from globalVariables.py

    returns
    -------
    x : =/=
        bandpass-noised image
    """
    if random.random() < p:
        a = random.randint(0, input_shape[0]//2)
        b = random.randint(a + 20, input_shape[0])
        x[a:b, :] += (np.random.sample((b - a, input_shape[1])).astype(np.float32) +
                      9) * x.mean() * noise_level * (np.random.sample() + 0.3)

    return x


def applyGaussianBlur(image, gauss_blur_limit, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur
    """
    GaussianBlur = A.GaussianBlur(blur_limit=gauss_blur_limit, p=p)
    image = GaussianBlur(image=image)['image']

    return image


def applyGlassBlur(image, glass_blur_maxdelta, glass_blur_iterations, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GlassBlur
    """
    GlassBlur = A.GlassBlur(max_delta=glass_blur_maxdelta,
                            iterations=glass_blur_iterations, p=p)
    image = GlassBlur(image=image)['image']

    return image


def applyRandomGamma(image, gamma_limit, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGamma
    """
    RandomGamma = A.RandomGamma(gamma_limit=gamma_limit, p=p)
    image = RandomGamma(image=image)['image']

    return image


def applySharpen(image, sharpen_alpha, sharpen_lightness, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Sharpen
    """
    Sharpen = A.Sharpen(alpha=sharpen_alpha, lightness=sharpen_lightness, p=p)
    image = Sharpen(image=image)['image']

    return image


def applyDownscaling(image, downscale_min, downscale_max, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Downscale
    """
    Downscale = A.Downscale(scale_min=downscale_min,
                            scale_max=downscale_max, p=p)
    image = Downscale(image=image)['image']

    return image


def applyEmboss(image, emboss_strength, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Emboss
    """
    Emboss = A.Emboss(strength=emboss_strength, p=p)
    image = Emboss(image=image)['image']

    return image

# GridDistortion is recommended to use with another permutation technique


def applyGridDistortion(image, p):

    GridDistortion = A.GridDistortion(p=p)
    image = GridDistortion(image=image)['image']

    return image

# OpticalDistortion is recommended to use with another permutation technique


def applyOpticalDistortion(image, distort_limit, shift_limit, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GridDistortion
    """
    OpticalDistortion = A.OpticalDistortion(
        distort_limit=distort_limit, shift_limit=shift_limit, p=p)
    image = OpticalDistortion(image=image)['image']

    return image


def applyInvertImage(image, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.InvertImg
    """
    InvertImage = A.InvertImg(p=p)
    image = InvertImage(image=image)['image']

    return image


def applyRotateImage(image, limit, p):
    """
    read from source https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.Rotate
    """
    Rotate = A.Rotate(limit=limit, p=p)
    image = Rotate(image=image)['image']

    return image


def classification_permutations(image, permutations):
    """
    returns a permutated image using a composition of permutations

    parameters
    ----------
    image : np.array
        input image

    permutations : list
        list of function permutations

    returns
    -------
    image : np.array
        permutated image
    """
    transformations = A.Compose([permutation for permutation in permutations])
    transformed = transformations(image=image)
    image = transformed['image']

    return image


def detection_permutations(image, bboxes, bbox_format, permutations):
    """
    returns a permutated image using a composition of permutations

    parameters
    ----------
    image : np.array
        input image

    bboxes :
        #TODO: add description

    bbox_format : 
        #TODO: add description

    permutations : list
        list of function permutations

    returns
    -------
    image : np.array
        permutated image

    bboxes : 
        #TODO: add description
    """

    # add class label
    for box in bboxes:
        box.append(1)

    transformations = A.Compose([permutation for permutation in permutations],
                                bbox_params=A.BboxParams(format=bbox_format))

    transformed = transformations(image=image, bboxes=bboxes)
    image = transformed['image']
    bboxes = transformed['bboxes']

    # transform bboxes format to Tensorflow API format and remove class label
    bboxes_tf = []
    for box in bboxes:
        box_tf = [box[1], box[0], box[3], box[2]]
        bboxes_tf.append(box_tf)

    return image, bboxes_tf
