import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax

import os
import numpy as np
import pandas as pd
import shutil
from tqdm.notebook import tqdm

os.chdir('V:\Git\spellbook')

from globalVariables import (
    INPUT_SHAPE, NUM_CLASSES, NUM_ADD_CLASSES, LABEL_IDX, LABEL_IDXS_ADD, FROM_LOGITS, 
    ARCMARGIN_M, ARCMARGIN_S)

from helpers import loadImage, loadNumpy, createOneHotVector, evaluateString
from preprocessFunctions import addColorChannels, kerasNormalize
from layers import ArcMarginProduct


def prepareImage(path, model_name, image_arr, unprocessed_image):

    if not unprocessed_image:

        image = loadNumpy(path)
    
    else:

        image = image_arr

    # normalization_function = tfimm.create_preprocessing(model_name, dtype="float32")

    normalization_function = kerasNormalize(model_name)

    image = normalization_function(image)
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)

    # image_inception = tf.keras.applications.inception_v3.preprocess_input(image)
    # image_inception = tf.convert_to_tensor(image_inception)
    # image_inception = tf.expand_dims(image_inception, axis=0)

    return image


def predictId():

    image_dir = 'projects/happywhale-2022/data/test_numpy_768/'
    image_names = os.listdir(image_dir)

    unprocessed_image = False
    predict_specie = False
    id_known = False
    
    model_name = 'EfficientNetB5'
    model_species = load_model('projects/happywhale-2022/models/species_sparse_nofold/savedModel/')
    folds = ['1', '2', '3', '4']
    model_ids_1 = load_model('projects/happywhale-2022/models/ids_arcmargin_sparse_kfold/fold-1/savedModel/')
    model_ids_2 = load_model('projects/happywhale-2022/models/ids_arcmargin_sparse_kfold/fold-2/savedModel/')
    model_ids_3 = load_model('projects/happywhale-2022/models/ids_arcmargin_sparse_kfold/fold-3/savedModel/')
    model_ids_4 = load_model('projects/happywhale-2022/models/ids_arcmargin_sparse_kfold/fold-4/savedModel/')

    species_meta = pd.read_csv('projects/happywhale-2022/data/metadata/species_idxs.csv')
    individual_ids_meta = pd.read_csv('projects/happywhale-2022/data/metadata/individual_ids_idxs.csv')

    new_i_threshold = 0.6

    images_processed = 0
    prediction = []
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

        y_pred_species = model_species(image)
        specie_idx = np.argmax(y_pred_species)
        # onehot_species = np.zeros((26))
        # onehot_species[specie_idx] = 1
        onehot_species_tf = tf.convert_to_tensor(specie_idx, dtype=tf.float32)
        onehot_species_tf = tf.expand_dims(onehot_species_tf, axis=0)

        y_pred_fold_1 = model_ids_1([image, onehot_species_tf])
        y_pred_fold_2 = model_ids_2([image, onehot_species_tf])
        y_pred_fold_3 = model_ids_3([image, onehot_species_tf])
        y_pred_fold_4 = model_ids_4([image, onehot_species_tf])

        y_pred = 0.3 * y_pred_fold_1 + 0.3 * y_pred_fold_2 + 0.2 * y_pred_fold_3 + 0.2 * y_pred_fold_4

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
                individual_id = individual_id.replace('_DECIMAL', '')

            prediction_ids.append(individual_id)
        
        prediction_ids_string = ' '.join(prediction_ids)

        prediction.append([image_name.replace('.npy', ''), prediction_ids_string])

        images_processed += 1

        # if images_processed == 100:
        #     break
    
    prediction_meta = pd.DataFrame(prediction, columns=['image', 'predictions'])
    prediction_meta.to_csv(path_or_buf='projects/happywhale-2022/data/metadata/submissions/EfficientNetB5_EfficientNetB5_folds.csv', index=False)


def checkPredictions():

    submission = pd.read_csv('projects/happywhale-2022/data/metadata/submissions/EfficientNetB5_EfficientNetB7_15.csv')

    individual_ids_meta = pd.read_csv('projects/happywhale-2022/data/metadata/train_metadata.csv')
    i_ids_values = individual_ids_meta['individual_id'].values

    incorrect_predictions = []

    for idx, row in submission.iterrows():

        predictions = row['predictions']
        predictions = predictions.split(' ')
        
        for prediction in predictions:

            if prediction not in i_ids_values:
                incorrect_predictions.append(prediction)

    unique_ids = np.unique(incorrect_predictions)

    for unique_id in unique_ids:

        unique_id_decimal = unique_id + '_DECIMAL'

        if unique_id_decimal not in i_ids_values:

            print(unique_id_decimal)


def aa():

    model = efn.EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
    model.layers[0]._name = 'inp1'

    gap1=tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
    gap2=tf.keras.layers.GlobalAveragePooling2D()(model.layers[-5].output)
    gap3=tf.keras.layers.GlobalAveragePooling2D()(model.layers[-7].output)
    gap4=tf.keras.layers.GlobalAveragePooling2D()(model.layers[-13].output)

    def dropoutDense(x, drop, dense):
        x = tf.keras.layers.Dropout(drop)(x)
        return tf.keras.layers.Dense(dense)(x)

    x1 = dropoutDense(gap1, drop=0.3, dense=2560)
    x2 = dropoutDense(gap2, drop=0.3, dense=640)
    x3 = dropoutDense(gap3, drop=0.3, dense=640)
    x4 = dropoutDense(gap4, drop=0.3, dense=3840)

    embedding =  tf.concat([x1,x2,x3,x4], axis=1)

    input_features_layer = tf.keras.layers.Input(shape=(), name=('input_features_layer_1'))

    arc_margin = ArcMarginProduct(
        n_classes=NUM_CLASSES, s=ARCMARGIN_S, m=ARCMARGIN_M, name='head/arcmargin', dtype='float32')
    arc_margin_layer = arc_margin([embedding, input_features_layer])

    output = tf.keras.layers.Softmax(dtype='float32')(arc_margin_layer)

    model_embed = tf.keras.models.Model(inputs=model.input, outputs=embedding)
    model_arc = tf.keras.models.Model(inputs=[model.input, input_features_layer], outputs=[output])

    model_arc.load_weights('projects/happywhale-2022/models/effnetb7_loaded/effnetv1_b7_loss_0.h5')
