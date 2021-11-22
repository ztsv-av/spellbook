# SIIM-FISABIO-RSNA COVID-19 Detection

![covid-19](https://user-images.githubusercontent.com/73081144/142792695-ecfe8041-7933-4ddd-bece-a4ed91523782.png)

## Description

In this competition, we had to identify and localize COVID-19 abnormalities on chest radiographs.
In particular, we categorized the radiographs as negative for pneumonia, typical, indeterminate or atypical for COVID-19
and drew bounding boxes around the affected area in lungs.
The dataset consisted of different studies.
Each study corresponded to a single patient
and could contain single or multiple chest radiographs (DICOM files) corresponding to that patient.
The same patient could have multiple affected areas with different kind of severity,
meaning that it was a multi-label classification task
(even though I say it was a multi-label classification, the provided train metadata had only one label per COVID-19 study,
but the competition host specifically pointed out that the test data had multiple labels for most of the radiographs).
The data also contained metadata with COVID-19 categories and bounding boxes for every DICOM file.

&nbsp;

## Data Preprocessing

Before feeding the data to the classification and localization models, we:
- converted DICOM files to numpy arrays;
- equalized height and width of an image and resize it to the desired shape (as well as corresponding bounding boxes);
- min-maxed normalize the numpy array
- converted the image from grayscale to RGB for proper training.

&nbsp;

## Data Augmentation

For data augmentation we used the following techniques from [albumentations](https://albumentations.ai/) library:
- RandomGamma;
- HorizontalFlip;
- GaussianBlur
- GlassBlur;
- Sharpen;
- Emboss;
- GridDistortion;
- OpticalDistortion;
- Rotate.

&nbsp;

## Training

### Models

For localization we used EfficientDet D0 512x512 downloaded from [TensorFlow Hub](https://tfhub.dev/) and trained it on COVID-19 dataset.
For classification we trained multiple models taken from tf.keras.applications library (such as EfficientNetB0, EfficientNetB1, EfficientNetB2, DenseNet121, ...)

### Loss

Because this competition is a multi-label classification and due to label imbalance we used binary focal loss.

### Optimizer

We used Adam optimizer with step-based learning rate decay scheduler.

&nbsp;

## Prediction

To localize bounding boxes, we used trained EfficientDet D0 model.
To classify a radiograph, we used an ensemble of trained keras models (an average over label predictions from different models).
