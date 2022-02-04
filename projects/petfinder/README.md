# PetFinder.my - Pawpularity Contest

![image](https://user-images.githubusercontent.com/73081144/138211735-05666f5d-9638-4150-afa8-b5b283716d25.png)

## Description

In this competition, we analyze raw images and metadata to predict the “Pawpularity” score of pet photos (regression challenge). We train and test our models on PetFinder.my's pet profiles.

Initially, we thought of using only provided data to predict Pawpularity score. We were given pet images and tabular meta features.
However, after running correlation tests between metadata and Pawpularity score, we noticed that there was not any correlation at all. So, we decided to drop given tabular data.

Still, we needed some metadata that would help predict Pawpularity score, such as breed of animal, age, type (cat or dog), fur color, fur length, animal size, etc. Surely, those features would grately affect Pawpularity score (atleats, we thought so).

We came up with a following plan:
1. Download datasets that contained metadata with some of the nessecary features.
2. Train multiple different models that would predict those features (one model - one feature)
3. Use those models to predict features for Petfinder data
4. Train final model to predict Pawpularity score using given images and predicted additional features.

Datasets that we used:
- [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction/data) (used to train Age and Color feature model)
- [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) (used to train Breed feature model)
- [Cat Breeds Dataset](https://www.kaggle.com/ma7555/cat-breeds-dataset) (needed serious data cleanup - a lot of wrongly classified images) (used to train Breed and Age features model)
- [Cats and Dogs Breeds](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset/) Classification Oxford Dataset (used to train Breed feature model)
- [The DogAge Dataset](https://www.kaggle.com/user164919/the-dogage-dataset) (used to train Age feature model)

For features models we used pretrained ImageNet InceptionV3 model with different head layers. Also, we partially unfroze some of the top layers, which increased fitting speed.
We decided to predict 4 main features: Type, Color, Breed, Age. For each of these features we had the following results:
- Type (2 classes): categorical accuracy - 99.7
- Color (7 classes): binary accuracy - 86.5
- Breed (167 classes, we predicted only the most prominent breed): categorical accuracy: fold 1 - 77.2, fold 2 - 77.3, fold 3 - 77.9, fold 4 - 77.5, fold 5 - 80.6
- Age (4 classses - baby, young, adult, senior): categorical accuracy - 80.8

After that, we used these models to predict those features on Petfinder images.
We tried to use the same InceptionV3 model to predict Pawpularity score. After the global pooling layer we concatenated additional features with extracted image features, passed them to multiple Fully Connected layers and to the final 1 neuron Dense layer. RMSE score was extremely high.
So, we decided to use a denoising autoencoder (the idea taken from here: [part of 9th place (denoising auto-encoder NN)](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88740)). The idea was to train the model to encode image features and additional features, apply swap noise, and decode additional features back. After fitting, we would concatenate last Fully Connected layers together (encoder, bottleneck and decoder layers), add new Fully Connected layers and the last 1 neuron Dense layer and retrain the model to predict Pawpularity score. Unfortunately, this idea was not successful either.

## Data Preprocessing

### Additional Data



## Data Augmentation

## Training

### Models

### Loss

### Optimizer

## Prediction
