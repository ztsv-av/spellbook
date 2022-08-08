# BirdCLEF 2022

![image](https://user-images.githubusercontent.com/73081144/183530394-2dcdcd35-17b4-4704-b444-8351669219c4.png)

## Description

In this competition, we automated the acoustic identification of birds in soundscape recordings with the help of deep learning models. 
We examined an acoustic dataset and build a classification model which categorized signals of interest (bird calls) in a recording.

The difference between BIRDCLEF-2021 and this competition is that in the test set of BIRDCLEF-2022 there are some bird categories that were not included in the training set. The task was to predict known 21 bird classes (not 152 as in the training set), and if the call is of an unknown bird, assign it to the 22nd class.


&nbsp;

## Data Preprocessing

Data preprocessing techniques include:
- removed audio from a recording where there is no bird call;
- add random background sounds;
- merge random recordings together to create more multi-label data;
- convert recordings to melspectograms.

&nbsp; 

## Data Augmentation

For data augmentation, we added white and bandpass noise to the recording. Also, we raised melspectograms to the power of 3 and amplified the signal.

&nbsp; 

## Training

### Models

We used pretrained imagenet keras EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB5 and Xception models. Trained them on 152 bird classes and used these models to predict 22 bird classes (which were included in the above mentioned 152 classes.)

### Loss

We used binary focal loss.

### Optimizer

We used Adam optimizer and reduced learning rate by a factor of 2 on plateau.

&nbsp; 

## Prediction

We used ensemble prediction, but didn't apply any weight to any of the models' predictions.

&nbsp; 

## Improvement

The main reason why this approach wasn't enough to reach top 100 is that we used Tensorflow pipeline. With same architecture the results are much more better with PyTorch approach instead of the Tensorflow.
