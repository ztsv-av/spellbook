## Exoplanet Hunting Challenge
---

&nbsp; 

## Description

The data describe the change in flux (light intensity) of several thousand stars. Each star has a binary label of 2 or 1. 2 indicated that that the star is confirmed to have at least one exoplanet in orbit; some observations are in fact multi-planet systems.

As you can imagine, planets themselves do not emit light, but the stars that they orbit do. If said star is watched over several months or years, there may be a regular 'dimming' of the flux (the light intensity). This is evidence that there may be an orbiting body around the star; such a star could be considered to be a 'candidate' system. Further study of our candidate system, for example by a satellite that captures light at a different wavelength, could solidify the belief that the candidate can in fact be 'confirmed'.

&nbsp; 

## Data Preprocessing

To deal with outliers, we simply replace each maximum and minimum values with average for each FLUX record.

&nbsp; 

## Data Augmentation

The peculiarity of this problem is that it has highly imbalanced training and test datasets. To deal with the problem, the SMOTE (oversampling) technique combined with random undersampling is used as per ([Chawla et al., 2002](https://arxiv.org/pdf/1106.1813.pdf)).

&nbsp; 

## Training

### Models

The following CNN architecture is used to achieve **perfect score (100%)**:

* **Input layer**;
* **1D convolutional layer**, consisting of 10 2x2 filters, L2 regularization and RELU activation function;
* **1D max pooling layer**, window size - 2x2, stride - 2;
* **Dropout** with 20% probability;
* **Fully connected layer** with 32 neurons and RELU activation function;
* **Dropout** with 40% probability;
* **Fully connected layer** with 18 neurons and RELU activation function;
* **Output layer** with sigmoid function.

As it is suggested in papers ([Hinton et al., 2021](https://arxiv.org/pdf/1207.0580.pdf), [Park & Kwak, 2016](http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf)), we use 20% dropout after 1D CONV layers and 40-50% dropout after fully connected layers. For training, we initialized batch size to 64 and number of epochs to 30. Also, we use exponential decay and early stopping to prevent non-convergence and overfitting.

### Loss

Default binary-crossentropy loss function is used.

### Optimizer

Default adam optimizer is used.

&nbsp; 

## Prediction

Training on GPU involves a certain degree of randomness. On average, this model achieves a perfect score on y=1 (star has an exoplanet) around 20 times for every 200 simulations.

![image](https://user-images.githubusercontent.com/73081144/114321610-f0e3b380-9ad8-11eb-91b2-38526202d29d.png)