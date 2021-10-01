# Templates

This file contains templates and formats on how to structure and document

- [Python objects](#python-object-documentation) such as functions and classes,
- [repo's directories](#repo-directory-structure) for proper render on GitHub,
- [project's readme](#project-readme-template) to grasp the idea behind
  the work done, and
- [object detection metadata](#object-detection-metadata) for object detection
  fit.

## Python Object Documentation

The general template to describe Python functions and classes is as follows:

```py
def function(array, a=5.0,...):

    """
    multiplies numpy array by constant

    parameters
    ----------
    a : float, default is 5.0
        input constant

    array : ndarray
        input numpy array

    returns
    -------
    multiplied_array : ndarray
        product of numpy array and constant
    """

    multiplied_array = array * a

    return multiplied_array
```

## Repo Directory Structure

The directories comply with the following template:

```md
Each project directory consists of following folders:

| Folder         | Description                                           |
| -------------- | ----------------------------------------------------- |
| `datasets`     | Project datasets and metadata files                   |
| `training`     | Trained models' weights and fit logs                  |
| `best_weights` | Best models' weights picked from `training` directory |
| `predictions`  | Predictions for test dataset                          |

and the following `.py` scripts:

| Script                 | Description                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `dataPreprocessing.py` | Preprocesses data for fitting (preprocessed data is saved to `datasets` directory) |
| `prediction.py`        | Prepares and saves predictions into `predictions` directory                        |
```

## Project `README` Template

The project's `README.md` complies with the following template:

```md
# Project Name

## Description

## Data Preprocessing

## Data Augmentation

## Training

### Models

### Loss

### Optimizer

## Prediction
```

## Object Detection Metadata

The `.csv` metadata complies with the following format:

```md
- first column must be named `filenames` and contain strings as names of the images (does not matter if filenames have file extension at the end);
- second column is `bboxes` and has a list of lists of bounding boxes which are stored in this format: `ymin, ymax, xmin, xmax`. Note: each bounding box must be NORMALIZED;
- (optional) third column is `classes` which contains integers as classes for each image.
```
