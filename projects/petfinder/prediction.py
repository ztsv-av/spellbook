import tensorflow as tf
from tensorflow.nn import softmax
from tensorflow.math import sigmoid
from tensorflow.keras.models import load_model

import os
import numpy as np
import pandas as pd
import shutil
from tqdm.notebook import tqdm

os.chdir('V:\Git\spellbook')

from globalVariables import (
    INPUT_SHAPE, MODEL_POOLING, INITIAL_DROPOUT, DO_BATCH_NORM, DROP_CONNECT_RATE, CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, FC_LAYERS, DROPOUT_RATES, DO_PREDICTIONS, NUM_CLASSES, OUTPUT_ACTIVATION)

from models import buildClassificationImageNetModel, buildAutoencoderPetfinder
from helpers import loadNumpy, getFeaturesFromPath, evaluateString, getFullPaths, visualizeImageBox, saveNumpyArray
from preprocessFunctions import minMaxNormalizeNumpy


# input_image_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_image')

# classification_model = buildClassificationImageNetModel(
#     input_image_layer, 'Xception', tf.keras.applications.Xception,
#     MODEL_POOLING, DROP_CONNECT_RATE, INITIAL_DROPOUT, FC_LAYERS, DROPOUT_RATES,
#     NUM_CLASSES, OUTPUT_ACTIVATION, trainable=False, do_predictions=True)

# model = buildAutoencoderPetfinder(
#     imageFeatureExtractor.output.shape[1], NUM_FEATURES, IMAGE_FEATURE_EXTRACTOR_FC, AUTOENCODER_FC)


def prepareImageLabels(path, metadata):

    image = loadNumpy(path)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)

    # label = getFeaturesFromPath(path, METADATA, ID_COLUMN, FEATURE_COLUMN, False)
    # label = evaluateString(label)
    # label = tf.convert_to_tensor(label)
    # label = tf.expand_dims(label, axis=0)

    return image, None


def extractFeatures():

    image_dir = 'projects/petfinder/data/petfinder-images-preprocessed/'

    save_features_dir = 'projects/petfinder/data/petfinder-images-features/'

    input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data')
    input_layers = [input_data_layer]

    model_name = 'InceptionV3'
    model_imagenet = tf.keras.applications.InceptionV3

    classification_model = buildClassificationImageNetModel(
        input_layers, model_name, model_imagenet,
        MODEL_POOLING, DROP_CONNECT_RATE, False, None, False, False, None, None,
        NUM_CLASSES, OUTPUT_ACTIVATION, trainable=False, do_predictions=False)

    for image_name in tqdm(os.listdir(image_dir), desc='Images Processed'):

        image_path = image_dir + image_name

        image, _ = prepareImageLabels(image_path, None)

        extracted_features = classification_model(image)
        extracted_features = extracted_features[0].numpy()

        save_features_path = save_features_dir + image_name

        saveNumpyArray(extracted_features, save_features_path)


def predictBreed():

    # DON'T FORGET TO APPLY PROPER NORMALIZATION FUNCTION

    petfinder_images_dir = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed/'
    petfinder_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-preprocessed-metadata.csv')
    petfinder_age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-age-3classes-preprocessed.csv')
    
    dogs_images_dir = 'projects/petfinder/petfinder-previous/data/additional-data/dogs-age-preprocessed/'
    dogs_age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/dogs-age.csv')
    
    cats_images_dir = 'projects/petfinder/petfinder-previous/data/additional-data/cats-age-preprocessed/'
    cats_breed_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-full.csv')
    cats_age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/cats-age.csv')

    breed_fur_size_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-fur-size-preprocessed.csv')

    breed_type_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-type.csv')
    incorrect_breed_type_dir = 'projects/petfinder/petfinder-previous/data/additional-data/incorrect-breed-type/'
    incorrect_breed_type_list = []

    age_dir = 'projects/petfinder/petfinder-previous/data/images-age-preprocessed/'

    model_color = load_model('projects/petfinder/petfinder-previous/models/color/savedModel/', compile=False)

    model_fold1 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-1/savedModel/', compile=False)
    model_fold2 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-2/savedModel/', compile=False)
    model_fold3 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-3/savedModel/', compile=False)
    model_fold4 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-4/savedModel/', compile=False)
    model_fold5 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-5/savedModel/', compile=False)

    models_breed = [model_fold1, model_fold2, model_fold3, model_fold4, model_fold5]

    age_list = []

    for image_name in tqdm(os.listdir(petfinder_images_dir), desc='Petfinder'):

        image_path = petfinder_images_dir + image_name

        image, _ = prepareImageLabels(image_path, None)

        animal_type_petfinder = petfinder_meta[petfinder_meta['id'] == image_name.split('_')[0]]['Type'].values[0]
        if animal_type_petfinder == '[1, 0]':
            animal_type = [0, 1]
        elif animal_type_petfinder == '[0, 1]':
            animal_type = [1, 0]
        else:
            print('Something wrong with Petfinder "Type" ! ! !')
        animal_type_tf = tf.convert_to_tensor(animal_type)
        animal_type_tf = tf.expand_dims(animal_type_tf, axis=0)

        a_color = petfinder_meta[petfinder_meta['id']== image_name.split('_')[0]]['Color'].values[0]
        a_color = evaluateString(a_color)
        prediction_color_tf = tf.convert_to_tensor(a_color)
        prediction_color_tf = tf.expand_dims(prediction_color_tf, axis=0)

        prediction_breed_ensemble = np.zeros((167))

        for model in models_breed:

            breed_input = [image, animal_type_tf, prediction_color_tf]

            prediction_breed = model(breed_input)
            prediction_breed = softmax(prediction_breed[0].numpy())
            prediction_breed_ensemble += prediction_breed
        
        prediction_breed_ensemble = prediction_breed_ensemble / 5
        breed_id = np.argmax(prediction_breed_ensemble)
        prediction_breed = [0 for val in range(167)]
        prediction_breed[breed_id] = 1

        animal_fur = breed_fur_size_meta[breed_fur_size_meta['breed'] == str(prediction_breed)]['fur'].values[0]
        animal_size = breed_fur_size_meta[breed_fur_size_meta['breed'] == str(prediction_breed)]['size'].values[0]

        animal_age = petfinder_age_meta[petfinder_age_meta['id'] == image_name]['age'].values[0]

        age_list.append([image_name, animal_type, a_color, prediction_breed, animal_fur, animal_size, animal_age])

        correct_breed_type = evaluateString(breed_type_meta[breed_type_meta['breed'] == str(prediction_breed)]['type'].values[0])
        if correct_breed_type != animal_type:
            incorrect_breed_type_list.append([image_name, prediction_breed, animal_type, correct_breed_type])
            shutil.copy(image_path, incorrect_breed_type_dir + image_name)
        else:
            shutil.copy(image_path, age_dir + image_name)

    for image_name in tqdm(os.listdir(dogs_images_dir), desc='Dogs'):

        image_path = dogs_images_dir + image_name

        image, _ = prepareImageLabels(image_path, None)

        animal_type = [0, 1]
        animal_type_tf = tf.convert_to_tensor(animal_type)
        animal_type_tf = tf.expand_dims(animal_type_tf, axis=0)

        prediction_color = model_color(image)
        prediction_color = sigmoid(prediction_color[0].numpy())
        prediction_color = [1 if color >= 0.5 else 0 for color in prediction_color]
        prediction_color_tf = tf.convert_to_tensor(prediction_color)
        prediction_color_tf = tf.expand_dims(prediction_color_tf, axis=0)

        prediction_breed_ensemble = np.zeros((167))

        for model in models_breed:

            breed_input = [image, animal_type_tf, prediction_color_tf]

            prediction_breed = model(breed_input)
            prediction_breed = softmax(prediction_breed[0].numpy())
            prediction_breed_ensemble += prediction_breed
        
        prediction_breed_ensemble = prediction_breed_ensemble / 5
        breed_id = np.argmax(prediction_breed_ensemble)
        prediction_breed = [0 for val in range(167)]
        prediction_breed[breed_id] = 1

        animal_fur = breed_fur_size_meta[breed_fur_size_meta['breed'] == str(prediction_breed)]['fur'].values[0]
        animal_size = breed_fur_size_meta[breed_fur_size_meta['breed'] == str(prediction_breed)]['size'].values[0]

        animal_age = dogs_age_meta[dogs_age_meta['id'] == image_name]['age'].values[0]
        animal_age = evaluateString(animal_age)
        animal_age.insert(0, 0)

        if animal_age == [1, 0, 0, 0]:
            correct_age = [1, 0, 0]
        elif animal_age == [0, 1, 0, 0]:
            correct_age = [0, 1, 0]
        else:
            correct_age = [0, 0, 1]

        age_list.append([image_name, animal_type, prediction_color, prediction_breed, animal_fur, animal_size, correct_age])

        correct_breed_type = evaluateString(breed_type_meta[breed_type_meta['breed'] == str(prediction_breed)]['type'].values[0])
        if correct_breed_type != animal_type:
            incorrect_breed_type_list.append([image_name, prediction_breed, animal_type, correct_breed_type])
            shutil.copy(image_path, incorrect_breed_type_dir + image_name)
        else:
            shutil.copy(image_path, age_dir + image_name)

    for image_name in tqdm(os.listdir(cats_images_dir), desc='Cats'):

        image_path = cats_images_dir + image_name

        image, _ = prepareImageLabels(image_path, None)

        animal_type = cats_breed_meta[cats_breed_meta['id'] == image_name]['type'].values[0]
        animal_breed = cats_breed_meta[cats_breed_meta['id'] == image_name]['breed'].values[0]    
        animal_fur = cats_breed_meta[cats_breed_meta['id'] == image_name]['fur'].values[0]  
        animal_size = cats_breed_meta[cats_breed_meta['id'] == image_name]['size'].values[0]  
        animal_age = cats_age_meta[cats_age_meta['id'] == image_name]['age'].values[0]

        if animal_age == [1, 0, 0, 0]:
            correct_age = [1, 0, 0]
        elif animal_age == [0, 1, 0, 0]:
            correct_age = [0, 1, 0]
        else:
            correct_age = [0, 0, 1]

        prediction_color = model_color(image)
        prediction_color = sigmoid(prediction_color[0].numpy())
        prediction_color = [1 if color >= 0.5 else 0 for color in prediction_color]

        age_list.append([image_name, animal_type, prediction_color, animal_breed, animal_fur, animal_size, correct_age])

        shutil.copy(image_path, age_dir + image_name)

    incorrect_breed_type_meta = pd.DataFrame(incorrect_breed_type_list, columns=['id', 'breed', 'type','predicted-type'])
    incorrect_breed_type_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/incorrect-predicted-breed-type.csv', index=False)

    age_meta = pd.DataFrame(age_list, columns=['id', 'type', 'color','breed', 'fur', 'size', 'age'])
    age_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/age-full-last-predicted-breeds.csv', index=False)


def predictColor():

    # DON'T FORGET TO APPLY PROPER NORMALIZATION FUNCTION

    model_color = load_model('projects/petfinder/petfinder-previous/models/color/savedModel/', compile=False)

    images_dir = 'projects/petfinder/petfinder-previous/data/images-breeds-preprocessed/'
    breed_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-full.csv')

    color_meta_list = []

    for image_name in tqdm(os.listdir(images_dir), desc='Images Predicted'):

        image_path = images_dir + image_name

        image, _ = prepareImageLabels(image_path, None)

        prediction_color = model_color(image)
        prediction_color = sigmoid(prediction_color)[0].numpy()
        image_color = [1 if color >= 0.5 else 0 for color in prediction_color]

        a_type = breed_meta[breed_meta['id'] == image_name]['type'].values[0]
        a_breed = breed_meta[breed_meta['id'] == image_name]['breed'].values[0]
        a_fur = breed_meta[breed_meta['id'] == image_name]['fur'].values[0]
        a_size = breed_meta[breed_meta['id'] == image_name]['size'].values[0]

        color_meta_list.append([image_name, a_type, a_fur, a_size, image_color, a_breed])
    
    color_meta = pd.DataFrame(color_meta_list, columns=['id', 'type', 'fur', 'size', 'color', 'breed'])
    color_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/type-fur-size-color-breed.csv', index=False)


def predictAge():

    model_age = load_model('projects/petfinder/petfinder-previous/models/age/savedModel/', compile=False)

    images_dir = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed/'
    images_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-preprocessed-metadata.csv')
    breeds_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-breeds.csv')
    breed_fur_size_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-fur-size-preprocessed.csv')

    predicted_age_list = []

    for image_name in tqdm(os.listdir(images_dir), desc='Images Predicted'):

        petfinder_age = images_meta[images_meta['id'] == image_name.split('_')[0]]['Age'].values[0]

        if petfinder_age <= 3:

            petfinder_correct_age = [1, 0, 0]

            predicted_age_list.append([image_name, petfinder_correct_age])
        
        else:

            image_path = images_dir + image_name

            a_breed = breeds_meta[breeds_meta['id'] == image_name]['breed'].values[0]
            a_breed_int = evaluateString(breeds_meta[breeds_meta['id'] == image_name]['breed'].values[0])
            a_breed_tf = tf.convert_to_tensor(a_breed_int, dtype=tf.float32)
            a_breed_tf = tf.expand_dims(a_breed_tf, axis=0)

            a_fur = evaluateString(breed_fur_size_meta[breed_fur_size_meta['breed'] == a_breed]['fur'].values[0])
            a_fur_tf = tf.convert_to_tensor(a_fur, dtype=tf.float32)
            a_fur_tf = tf.expand_dims(a_fur_tf, axis=0)

            a_size = evaluateString(breed_fur_size_meta[breed_fur_size_meta['breed'] == a_breed]['size'].values[0])
            a_size_tf = tf.convert_to_tensor(a_size, dtype=tf.float32)
            a_size_tf = tf.expand_dims(a_size_tf, axis=0)

            image, _ = prepareImageLabels(image_path, None)

            age_input = [image, a_breed_tf, a_fur_tf, a_size_tf]

            prediction_age = model_age(age_input)
            prediction_age = softmax(prediction_age)[0].numpy()

            age_idx = np.argmax(prediction_age)

            one_hot_age = [0, 0, 0]
            one_hot_age[age_idx] = 1

            predicted_age_list.append([image_name, one_hot_age])
    
    predicted_age_meta = pd.DataFrame(predicted_age_list, columns=['id', 'age'])
    predicted_age_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/petfinder-predicted-age.csv', index=False)


def predictPetfinder():

    images_dir = 'projects/petfinder/data/petfinder-images-preprocessed/'

    petfinder_meta = pd.read_csv('projects/petfinder/data/metadata/petfinder-metadata.csv')

    model_type = load_model('projects/petfinder/petfinder-previous/models/type/savedModel/', compile=False)

    model_color = load_model('projects/petfinder/petfinder-previous/models/color/savedModel/', compile=False)

    model_breed_fold1 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-1/savedModel/', compile=False)
    model_breed_fold2 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-2/savedModel/', compile=False)
    model_breed_fold3 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-3/savedModel/', compile=False)
    model_breed_fold4 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-4/savedModel/', compile=False)
    model_breed_fold5 = load_model('projects/petfinder/petfinder-previous/models/breed/fold-5/savedModel/', compile=False)
    models_breed = [model_breed_fold1, model_breed_fold2, model_breed_fold3, model_breed_fold4, model_breed_fold5]

    model_age = load_model('projects/petfinder/petfinder-previous/models/age/savedModel/', compile=False)

    breed_fur_size_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-fur-size-preprocessed.csv')

    preprocessed_petfinder_meta_list = []

    for image_name in tqdm(os.listdir(images_dir), desc='Images Predicted'):

        image_path = images_dir + image_name

        image, _ = prepareImageLabels(image_path, None)

        prediction_type = model_type(image)
        prediction_type = softmax(prediction_type[0].numpy())
        type_idx = np.argmax(prediction_type)
        a_type = [0, 0]
        a_type[type_idx] = 1
        prediction_type_tf = tf.convert_to_tensor(a_type)
        prediction_type_tf = tf.expand_dims(prediction_type_tf, axis=0)

        prediction_color = model_color(image)
        prediction_color = sigmoid(prediction_color[0].numpy())
        a_color = [1 if color >= 0.5 else 0 for color in prediction_color]
        prediction_color_tf = tf.convert_to_tensor(a_color)
        prediction_color_tf = tf.expand_dims(prediction_color_tf, axis=0)

        prediction_breed_ensemble = np.zeros((167))

        for model in models_breed:

            breed_input = [image, prediction_type_tf, prediction_color_tf]

            prediction_breed = model(breed_input)
            prediction_breed = softmax(prediction_breed[0].numpy())
            prediction_breed_ensemble += prediction_breed
        
        prediction_breed_ensemble = prediction_breed_ensemble / 5
        breed_idx = np.argmax(prediction_breed_ensemble)
        a_breed = [0 for val in range(167)]
        a_breed[breed_idx] = 1
        prediction_breed_tf = tf.convert_to_tensor(a_breed)
        prediction_breed_tf = tf.expand_dims(prediction_breed_tf, axis=0)

        a_fur = evaluateString(breed_fur_size_meta[breed_fur_size_meta['breed'] == str(a_breed)]['fur'].values[0])
        fur_tf = tf.convert_to_tensor(a_fur)
        fur_tf = tf.expand_dims(fur_tf, axis=0)

        a_size = evaluateString(breed_fur_size_meta[breed_fur_size_meta['breed'] == str(a_breed)]['size'].values[0])
        size_tf = tf.convert_to_tensor(a_size)
        size_tf = tf.expand_dims(size_tf, axis=0)

        age_input = [image, prediction_type_tf, prediction_breed_tf, fur_tf, size_tf]
        prediction_age = model_age(age_input)
        prediction_age = softmax(prediction_age[0].numpy())
        a_age = [0, 0, 0]
        age_idx = np.argmax(prediction_age)
        a_age[age_idx] = 1

        a_popularity = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Pawpularity'].values[0]

        a_focus = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Subject Focus'].values[0]
        if a_focus == 0:
            a_focus = [1, 0]
        else:
            a_focus = [0, 1]

        a_eyes = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Eyes'].values[0]
        if a_eyes == 0:
            a_eyes = [1, 0]
        else:
            a_eyes = [0, 1]

        a_face = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Face'].values[0]
        if a_face == 0:
            a_face = [1, 0]
        else:
            a_face = [0, 1]

        a_near = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Near'].values[0]
        if a_near == 0:
            a_near = [1, 0]
        else:
            a_near = [0, 1]

        a_action = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Action'].values[0]
        if a_action == 0:
            a_action = [1, 0]
        else:
            a_action = [0, 1]

        a_accessory = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Accessory'].values[0]
        if a_accessory == 0:
            a_accessory = [1, 0]
        else:
            a_accessory = [0, 1]

        a_group = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Group'].values[0]
        if a_group == 0:
            a_group = [1, 0]
        else:
            a_group = [0, 1]

        a_collage = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Collage'].values[0]
        if a_collage == 0:
            a_collage = [1, 0]
        else:
            a_collage = [0, 1]

        a_human = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Human'].values[0]
        if a_human == 0:
            a_human = [1, 0]
        else:
            a_human = [0, 1]

        a_occlusion = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Occlusion'].values[0]
        if a_occlusion == 0:
            a_occlusion = [1, 0]
        else:
            a_occlusion = [0, 1]

        a_info = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Info'].values[0]
        if a_info == 0:
            a_info = [1, 0]
        else:
            a_info = [0, 1]

        a_blur = petfinder_meta[petfinder_meta['Id'] == image_name.split('_')[0]]['Blur'].values[0]
        if a_blur == 0:
            a_blur = [1, 0]
        else:
            a_blur = [0, 1]

        preprocessed_petfinder_meta_list.append([image_name, a_type, a_color, a_breed, a_fur, a_size, a_age, a_focus, a_eyes, a_face, a_near, a_action, a_accessory, a_group, a_collage, a_human, a_occlusion, a_info, a_blur, a_popularity])

    preprocessed_petfinder_meta = pd.DataFrame(preprocessed_petfinder_meta_list, columns=['id', 'type', 'color', 'breed', 'fur', 'size', 'age', 'focus', 'eyes', 'face', 'near', 'action', 'accessory', 'group', 'collage', 'human', 'occlusion', 'info', 'blur', 'popularity'])
    preprocessed_petfinder_meta.to_csv(path_or_buf='projects/petfinder/data/metadata/petfinder-preprocessed-metadata.csv', index=False)



        
