# BirdCLEF 2021 - Birdcall Identification

![birdclef](https://user-images.githubusercontent.com/73081144/142802087-fcebf012-f222-4d7c-9d5e-0b2fd7b77859.png)

## Description
- 
In this competition, we automated the acoustic identification of birds in soundscape recordings with the help of deep learning models. 
We examined an acoustic dataset and build a classification model which categorized signals of interest (bird calls) in a recording.


&nbsp;

## Data Preprocessing

Data preprocessing techniques include:
- removed audio from a recording where there is no bird call;
- add random background sounds;
- merge random recordings together to create more multi-label data;
- convert recordings to melspectograms.

&nbsp; 

## Data Augmentation

For data augmentation, we added background forest and rain sounds,
as well as white and bandpass noise.

&nbsp; 

## Training

### Models

We used pretrained keras DenseNet121 and trained it on the BirdCLEF dataset.

### Loss

We used binary crossentropy loss.

### Optimizer

We used Adam optimizer with step-based learning rate decay scheduler.

&nbsp; 

## Prediction

We didn't use ensemble prediction due to time limitation and lack of experience. If we had used it, we could have easily gotten in top 100 in private leaderboard.

