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

Each project directory consists of following folders:

| Folder         | Description                                           |
| -------------- | ----------------------------------------------------- |
| `data`         | Project datasets and metadata files                   |
| `training`     | Trained models' weights and fit logs                  |
| `best_weights` | Best models' weights picked from `training` directory |

and the following `.py` scripts:

| Script                 | Description                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `dataPreprocessing.py` | Preprocesses data for fitting (preprocessed data is saved to `data` directory)     |
| `prediction.py`        | Code for predictions                                                               |
| `train.py`             | (Optional) Contains a project specific train loop                                  |

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

- Column `A`:
  - must be named `filenames`, and
  - contain strings as names of the images (does not matter if filenames have file extension at the end),
- Column `B`:
  - must be named `bboxes`,
  - has a list (`[]`) of lists of **normalized** bounding boxes the format `[ymin, xmin, ymax, xmax]`, and
- _optional_ column `C`:
  - is named `classes` and contains integers as classes for each image.
