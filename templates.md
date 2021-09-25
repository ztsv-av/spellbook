# Object Documentation

The general template to describe Python objects is as follows

```
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
# Project Directory File Architecture

```
Each project directory consists of following folders:

 - datasets - contains project datasets and metadata files
 - training - contains trained models' weights and training logs
 - best_weights - contains the best models' weights picked from `training` directory
 - predictions - contains predictions for *test* dataset

and `.py` files:

 - `dataPreprocessing.py` - script to transform data into general representation appropriate for fitting (data is saved to `datasets` directory);
 - `prediction.py` - script to prepare and save predictions into `predictions` directory.
```

# Project Description

```
# Project Name

## Description
- 

&nbsp;

## Data Preprocessing

- 

&nbsp; 

## Data Augmentation

- 

&nbsp; 

## Training

### Models
- 

### Loss
- 

### Optimizer
- 

&nbsp; 

## Prediction
- 
```