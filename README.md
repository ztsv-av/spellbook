![spellbook](https://user-images.githubusercontent.com/73081144/134788873-c1401000-5fe5-45da-8902-96891c218d69.png)

# Spellbook

This repo aims to optimize usability of DL techniques by implementing a
standardized code piece suitable to various DL tasks such as classification,
object detection, regression, etc.

By directly importing the code piece the learning canvas may be built rather
fast and allow engineers to focus on task-specific issues rather than
cross-task, reusable actions.

The goal is to only change global variables contained in the globalVariables.py file
and start the whole data engineering and model/s training pipeline in either 
magicClassification.py or magicDetection.py files. Data preprocessing is done
differently for each project/competition beforehand.

## Spellbook Contents

The table below contains the descriptions for each of the scripts this repo
provides:

| Script                      | Description                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `callbacks.py`              | Custom callbacks for `Keras` library                                                                  |
| `globalVariables.py`        | Global variables and parameters such as data shapes, learning rate decay, etc.                        |
| `helpers.py`                | Miscellaneous functions                                                                               |
| `layers.py`                 | Custom / Specific layers.                                                                             |
| `losses.py`                 | Custom loss functions                                                                                 |
| `magicClassification.py`    | Main wrapper used to fit given models with given parameters and data. Used for classification/regression tasks.                                                                                                                                                               |
| `magicDetection.py`         | Main wrapper used to fit given models with given parameters and data. Used for object localization tasks.                                                                                                                                                                     |
| `metrics.py`                | Custom metrics                                                                                        |
| `models.py`                 | Custom models as well as pre-trained `Tensorflow` models (such as _ImageNet_)                         |
| `optimizers.py`             | Custom optimizers                                                                                     |
| `permutationFunctions.py`   | Functions for data permutations                                                                       |
| `prepareTrainDataset.py`    | Functions to prepare datasets (initialization of `tf.data.Dataset` object with optional permutations) |
| `preprocessingFunctions.py` | Functions for data preprocessing                                                                      |
| `templates.py`              | Templates and formats on how to structure and document                                                |
| `train.py`                  | Collection of train steps and full complete train cycles                                              |
