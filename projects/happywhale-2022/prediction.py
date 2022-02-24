import tensorflow as tf
from tensorflow.keras.models import load_model
import tfimm

import os
import numpy as np
import pandas as pd
import shutil
from tqdm.notebook import tqdm

os.chdir('V:\Git\spellbook')

from globalVariables import (
    INPUT_SHAPE, NUM_CLASSES, NUM_ADD_CLASSES, ONEHOT_IDX, ONEHOT_IDXS_ADD)

from helpers import loadImage, loadNumpy, createOneHotVector
from preprocessFunctions import addColorChannels


def prepareImage(path, model_name, image_arr, unprocessed_image):

    if not unprocessed_image:

        image = loadNumpy(path)
    
    else:

        image = image_arr

    normalization_function = tfimm.create_preprocessing(model_name, dtype="float32")
    image = normalization_function(image)
    # image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)

    return image


def predictId():

    model_ids_path = 'projects/happywhale-2022/models/individuals_fl_nofold/ConvNext/20/savedModel/'
    model_ids = load_model(model_ids_path)
    model_name = 'convnext_base_384_in22ft1k'

    model_species_path = 'projects/happywhale-2022/models/species_fl_nofold/InceptionV3/22/savedModel/'
    model_species = load_model(model_species_path)

    species_meta = pd.read_csv('projects/happywhale-2022/data/metadata/species_idxs.csv')
    individual_ids_meta = pd.read_csv('projects/happywhale-2022/data/metadata/individual_ids_idxs.csv')

    image_dir = 'projects/happywhale-2022/data/test_numpy_384/'
    image_names = os.listdir(image_dir)

    unprocessed_image = False
    predict_specie = True
    id_known = False

    new_i_threshold = 0.2

    prediction = []

    images_processed = 0

    for image_name in tqdm(image_names, desc='Images Predicted'):

        image_path = image_dir + image_name

        if unprocessed_image:

            image = loadImage(image_path, image_type='uint8')

            if len(image.shape) == 2:

                image = addColorChannels(image, 3)

            image = tf.image.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
            image = tf.cast(image, tf.uint8).numpy()

            image = prepareImage(None, model_name, image, unprocessed_image)
        
        else:

            image = prepareImage(image_path, model_name, None, unprocessed_image)

        if predict_specie:

            y_pred_species = model_species(image)
            specie_idx = np.argmax(y_pred_species)
            onehot_species = np.zeros((26))
            onehot_species[specie_idx] = 1
            onehot_species_tf = tf.convert_to_tensor(onehot_species, dtype=tf.float32)
            onehot_species_tf = tf.expand_dims(onehot_species_tf, axis=0)
            
            # onehot_species_tf = tf.math.round(y_pred_species)
        
        else:

            onehot_species = createOneHotVector(image_path, ONEHOT_IDXS_ADD[0], NUM_ADD_CLASSES[0])
            onehot_species_tf = tf.convert_to_tensor(onehot_species, dtype=tf.float32)
            onehot_species_tf = tf.expand_dims(onehot_species_tf, axis=0)
            specie_true = species_meta[species_meta['idx'] == np.argmax(onehot_species)]['specie'].values[0]

        if id_known:

            y_true = createOneHotVector(image_path, ONEHOT_IDX, NUM_CLASSES)
            i_id_true = individual_ids_meta[individual_ids_meta['idx'] == np.argmax(y_true)]['individual_id'].values[0]

        y_pred = model_ids([image, onehot_species_tf])
        y_pred_top_5 = tf.math.top_k(y_pred, k=5, sorted=True, name=None)
        y_pred_top_5_values = y_pred_top_5.values.numpy()
        y_pred_top_5_idxs = y_pred_top_5.indices.numpy()

        prediction_ids = []
        new_individual_set = False

        for idx, individual_id_idx in enumerate(y_pred_top_5_idxs[0]):

            individual_id_value = y_pred_top_5_values[0][idx]

            if (individual_id_value < new_i_threshold) and (new_individual_set is False):

                individual_id = 'new_individual'
                new_individual_set = True
            
            else:

                individual_id = individual_ids_meta[individual_ids_meta['idx'] == individual_id_idx]['individual_id'].values[0]

            prediction_ids.append(individual_id)
        
        prediction_ids_string = ' '.join(prediction_ids)

        prediction.append([image_name.replace('.npy', ''), prediction_ids_string])

        images_processed += 1

        # if images_processed == 5:
        #     break
    
    prediction_meta = pd.DataFrame(prediction, columns=['image', 'predictions'])
    prediction_meta.to_csv(path_or_buf='projects/happywhale-2022/data/metadata/submission_convnext.csv', index=False)

