![1](https://user-images.githubusercontent.com/73081144/134788824-6da60687-3615-4b88-9589-a88225f72b40.png)

# Spellbook

This repo aims to optimize usability of DL techniques by implementing a standardized code piece suitable to various DL tasks such as classification, object detection, etc. By directly importing the code piece the learning canvas may be built rather fast and allow engineers to focus on task-specific issues rather than cross-task, reusable actions.

## File List

### `callbacks.py`

Custom callbacks

### `generatorInstance.py`

Initializes batches of features and targets by iterating through filenames (obsolete)

### `globalVariables.py`

Space where all used variables are stored

### `helpers.py`

Useful miscellaneous functions

### `losses.py`

Custom loss functions

### `magic.py`

Working file, used to initialize parameters and start training
 
### `metrics.py`

Custom metrics

### `models.py`

Custom models as well as downloaded ones, such as *ImageNet* models

### `optimizers.py`

Custom optimizers

### `permutationFunctions.py`

Data permutation functions

### `prepareTrainDataset.py`

Dataset preparation functions (initializes tf.data.Dataset object and does permutations)

### `preprocessData.py`

Prepares data for training (e.x. normalization, rescailing, resizing, etc.)

### `preprocessingFunctions.py`

Data preprocessing functions

### `train.py` 

Takes data as a parameter and trains a given model on it
