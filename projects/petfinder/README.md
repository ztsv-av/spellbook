# PetFinder.my - Pawpularity Contest

![image](https://user-images.githubusercontent.com/73081144/138211735-05666f5d-9638-4150-afa8-b5b283716d25.png)

## Description

In this competition, we analyze raw images and metadata to predict the “Pawpularity” score of pet photos (regression challenge). We train and test our models on PetFinder.my's pet profiles.

Initially, we thought of using only provided data to predict Pawpularity score. We were given pet images and tabular meta features.
However, after running correlation tests between metadata and Pawpularity score, we noticed that there was not any correlation at all. So, we decided to drop given tabular data.

Still, we needed some metadata that would help predict Pawpularity score, such as breed of animal, age, type (cat or dog), fur color, fur length, animal size, etc. Surely, those features  should grately affect Pawpularity score (atleats, we thought so).

We came up with a following plan:
1. Download datasets that contained metadata with some of the nessecary features.
2. Train multiple different models that would predict those features (one model - one feature)
3. Use those models to predict features for Petfinder data
4. Train final model to predict Pawpularity score using given images and predicted additional features.

Datasets that we used:
- [PetFinder.my - Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score/overview) (competition dataset)
- [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction/data) (used to train Age and Color feature model)
- [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) (used to train Breed feature model)
- [Cat Breeds Dataset](https://www.kaggle.com/ma7555/cat-breeds-dataset) (needed serious data cleanup - a lot of wrongly classified images) (used to train Breed feature model)
- [Cats and Dogs Breeds](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset/) Classification Oxford Dataset (used to train Breed feature model)
- [The DogAge Dataset](https://www.kaggle.com/user164919/the-dogage-dataset) (used to train Age feature model)

For features models we used pretrained ImageNet InceptionV3 model with different head layers. Also, we partially unfroze some of the top layers, which increased fitting speed and model generalization.
We decided to predict 4 main features: Type, Color, Breed, Age. For each of these features we had the following results:
- Type (2 classes): categorical accuracy - 99.7
- Color (7 classes): binary accuracy - 86.5
- Breed (167 classes, we predicted only the most prominent breed): categorical accuracy: fold 1 - 77.2, fold 2 - 77.3, fold 3 - 77.9, fold 4 - 77.5, fold 5 - 80.6
- Age (4 classses - baby, young, adult, senior): categorical accuracy - 80.8

After that, we used these models to predict features for Petfinder images.
We tried to use the same InceptionV3 model to predict Pawpularity score. After the global pooling layer we concatenated additional features with extracted image features, passed them to multiple Fully Connected layers and to the final 1 neuron Dense layer. RMSE score was extremely high.
So, we decided to use a denoising autoencoder (the idea taken from here: [part of 9th place (denoising auto-encoder NN)](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88740)). The idea was to train the model to encode image features and additional features, apply swap noise, and decode additional features back. After fitting, we would concatenate last Fully Connected layers together (encoder, bottleneck and decoder layers), add new Fully Connected layers and the last 1 neuron Dense layer and retrain the model to predict Pawpularity score. Unfortunately, this idea was not successful either - we did not have enough time to code swap noise technique. 

Now, we have another idea - to use transfer learning logic. In previous Petfinder competition we had to predict AdoptionSpeed, which we think correlates with Pawpularity in this competition. So, the plan is this:
 - use additional features models to predict lacking features for previous Petfinder data.
 - train a classification model to predict AdoptionSpeed, change last prediction layer and continue to train this model on new Petfinder data or 
 - train an autoencoder to restore additional features, change last layers so that the model solves classification task, that is predicts AdoptionSpeed. Then, change last prediction layer to fit regression task and continue training on new Petfinder data.

## Data Preprocessing

We resized all images to 256x256x3 and used inception_v3 normalization function.

### Additional Data

As stated before, we used a total of 6 datasets in this competition.

1. [PetFinder.my - Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score/overview) (competition dataset)

We did not use repeating images (images with the same animal id).

No additinal data preprocessing was used.

2. [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction/data) (used to train Age and Color feature model)

We did not use repeating images (images with the same animal id).

Age feature was transformed to 3 class one hot vector. We divided them in the following way: 
   - younger than 7 month - 'Baby',
   - younger than 13 month old - 'Young',
   - older than 12 month - 'Adult' and 'Senior'. 

Breed feature was transformed to 167 class one hot vector (initially there were 307 breeds, but we only took animals with matching breeds from Stanford Dogs, Cat Breeds, Cats and Dogs Breeds datasets.)

Color feature was transformed to 7 class one hot vector.

3. [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) (used to train Breed feature model)

Contains 120 dog breeds with around 200 images for each breed.

We made sure that names of the breeds were the same as in Cats and Dogs Breeds dataset.

1. [Cat Breeds Dataset](https://www.kaggle.com/ma7555/cat-breeds-dataset) (used to train Breed feature model)

This dataset needed serious data cleanup - a lot of wrongly classified images.

In total we had 42 cat breeds.

5. [Cats and Dogs Breeds](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset/) Classification Oxford Dataset (used to train Breed feature model)

This dataset contained 37 breeds of cats and dogs.

6. [The DogAge Dataset](https://www.kaggle.com/user164919/the-dogage-dataset) (used to train Age feature model)

There were 3 age classes - young, adult and senior. We combined adult and senior features together.

## Data Augmentation

For data permutation we used random gamma, horizontal flip, glass blur, sharpen, emboss, random brightness and contrast, and hue saturation with 0.25 probability for every permutation technique.

## Training

### Models

For image features extraction we used ImageNet Inception V3. For additional feature prediction we modified ImageNet model head layers, and for Pawpularity we used a denoising autoencoder architecture.

### Loss

Depending on the task, we used following losses: binary crossentropy, categorical crossentropy, rmse loss, focal loss.

### Optimizer

Adam optimizer was used

## Prediction
