import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import efficientnet.tfkeras as efn

import os
import numpy as np
import pandas as pd
import shutil
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors

os.chdir('V:/Git/spellbook')

from globalVariables import INPUT_SHAPE, NUM_CLASSES, ARCMARGIN_M, ARCMARGIN_S
from helpers import saveNumpy, loadNumpy, evaluateString
from preprocessFunctions import kerasNormalize, minMaxNormalizeNumpy
from layers import ArcMarginProduct


# VARIABLES
DATA_EMBEDS_DIR = 'projects/happywhale-2022/data/data_embeddings_768/b7/mean/'
TEST_DIR = 'projects/happywhale-2022/data/test_numpy_768/'
TEST_EMBEDS_SAVE_DIR = 'projects/happywhale-2022/data/test_embeddings_768/b7/'
MODEL_NAME = 'EfficientNetB7'
EMBED_MODELS_DIR = 'projects/happywhale-2022/models/ids_arcmargin_sparse_kfold/'
FOLDS = ['0', '1', '2', '3', '4']
KNN_NEIGHBORS = 1000
NEW_I_THRESHOLD = 0.6
SAMPLE_LIST = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00','956562ff2888']
SAVE_TEST_EMBEDS = True


def prepareImageB5(path, model_name):

    image = loadNumpy(path)

    normalization_function = kerasNormalize(model_name)
    image = normalization_function(image)

    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)

    return image


def prepareImageB7(path):

    image = loadNumpy(path)

    image = image / 255.0

    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)

    return image


def getEmbeddings(image_path, model, model_name):

    image = prepareImageB5(image_path, model_name)
    embeddings = model.predict(image, verbose=0)
    
    return embeddings


def loadEmbeddings(filenames, dir, folds):

    embeds = [loadNumpy(dir + fold + '/' + filename) for (filename, fold) in zip(filenames, folds)]

    return embeds


def saveEmbeddings(data_dir, model_name, embed_models_path, folds, embed_save_dir):

    filenames = os.listdir(data_dir)

    for fold in tqdm(folds):

        embed_model = load_model(embed_models_path + fold + '/')

        for filename in tqdm(filenames):

            image_path = data_dir + filename
            embeddings_save_path = embed_save_dir + fold + '/' + filename.replace('.npy', '') + '_' + fold + '.npy'

            image = prepareImageB5(image_path, model_name)
            embeddings = embed_model.predict(image, verbose=0)
            # embeddings = getEmbeddings(image_path, embed_model, model_name)
            saveNumpy(embeddings, embeddings_save_path)
        
        del embed_model


def saveEmbeddingsMean(data_dir, data_embeds_dir, save_embeds_mean_dir, folds):

    data_filenames = os.listdir(data_dir)

    for data_filename in data_filenames:

        embeds_filenames = [data_filename.replace('.npy', '') + '_' + fold + '.npy' for fold in folds]
        embeds = loadEmbeddings(embeds_filenames, data_embeds_dir, folds)

        embeds_mean = np.mean(embeds, axis=0)
        embeds_mean_savepath = save_embeds_mean_dir + data_filename
        saveNumpy(embeds_mean, embeds_mean_savepath)


def getId(filename):

    i_id = filename.split('_')[-2] # filename.split('_')[-1].replace('.npy', '')

    return i_id


def trainKNN(data_embeds_dir, knn_neighbors):

    data_embeds_filenames = os.listdir(data_embeds_dir)

    train_embeds = [loadNumpy(data_embeds_dir + filename)[0] for filename in data_embeds_filenames]
    
    knn = NearestNeighbors(n_neighbors=knn_neighbors, metric='cosine')
    knn.fit(train_embeds)

    return knn


def decodeSparseId(id_encodings, sparse_id):

    individual_id = id_encodings[id_encodings['idx'] == evaluateString(sparse_id)]['individual_id'].values[0]
    individual_id = individual_id.replace('_DECIMAL', '')

    return individual_id


def createArcModel():

    model = efn.EfficientNetB7(
        weights=None, include_top=False, input_shape=[INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
    model.layers[0]._name = 'inp1'
    model.trainable = False

    gap1 = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
    gap2 = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-5].output)
    gap3 = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-7].output)
    gap4 = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-13].output)

    embedding =  tf.concat([gap1, gap2, gap3, gap4], axis=1)

    embedding = tf.keras.layers.Dropout(0.3)(embedding)
    embedding = tf.keras.layers.Dense(2048)(embedding)

    # model_embed = tf.keras.models.Model(inputs=[model.input], outputs=[embedding])

    input_features_layer = tf.keras.layers.Input(shape=(), name=('input_features_layer_1'))

    arc_margin = ArcMarginProduct(
        n_classes=NUM_CLASSES, s=ARCMARGIN_S, m=ARCMARGIN_M, name='head/arcmargin', dtype='float32')
    arc_margin_layer = arc_margin([embedding, input_features_layer])

    output = tf.keras.layers.Softmax(dtype='float32')(arc_margin_layer)

    model_arc = tf.keras.models.Model(inputs=[model.input, input_features_layer], outputs=[output])

    return model_arc


def predictIds(data_embeds_dir=DATA_EMBEDS_DIR, folds=FOLDS, knn_neighbors=KNN_NEIGHBORS, test_dir=TEST_DIR, test_embeds_save_dir=TEST_EMBEDS_SAVE_DIR, embed_models_dir=EMBED_MODELS_DIR, new_i_threshold=NEW_I_THRESHOLD, sample_list=SAMPLE_LIST, save_test_embeds=SAVE_TEST_EMBEDS):

    id_encodings = pd.read_csv('projects/happywhale-2022/data/metadata/individual_ids_idxs.csv')

    data_filenames = os.listdir(data_embeds_dir)
    data_targets = [filename.split('_')[-1].replace('.npy', '') for filename in data_filenames]
    data_targets = np.asarray(data_targets)

    test_filenames = os.listdir(test_dir)

    models_arc = []
    for fold in FOLDS:
        model_arc = createArcModel()
        weights_path = 'projects/happywhale-2022/models/effnetb7_loaded/effnetv1_b7_loss_' + fold + '.h5'
        model_arc.load_weights(weights_path)
        models_arc.append(model_arc)
    embed_models = [tf.keras.models.Model(inputs=[model.inputs[0]], outputs=[model.layers[-4].output]) for model in models_arc]

    del model_arc
    del models_arc

    # knn = trainKNN(data_embeds_dir, knn_neighbors)

    train_embeds = [loadNumpy(data_embeds_dir + filename) for filename in data_filenames]
    knn = NearestNeighbors(n_neighbors=knn_neighbors, metric='cosine')
    knn.fit(train_embeds)

    test_df = []

    for i, filename in enumerate(tqdm(test_filenames)):

        test_image_path = test_dir + filename

        embeds = []

        for fold, embed_model in enumerate(embed_models):

            image = prepareImageB7(test_image_path)
            embed = embed_model(image)[0].numpy()
            # embed = getEmbeddings(test_image_path, embed_model, model_name)

            if save_test_embeds:
                saveNumpy(embed, test_embeds_save_dir + str(fold) + '/' + filename)
            embeds.append(embed)

        embeds_mean = np.mean(embeds, axis=0)
        if save_test_embeds:
            saveNumpy(embeds_mean, test_embeds_save_dir + 'mean' + '/' + filename)

        embeds_mean = np.expand_dims(embeds_mean, axis=0)

        distances, idxs = knn.kneighbors(embeds_mean, knn_neighbors, return_distance=True)
        distances, idxs = distances[0], idxs[0]
        targets = data_targets[idxs]

        image_knn_data = pd.DataFrame(np.stack([targets, distances], axis=1), columns=['target', 'distances'])
        image_knn_data['id'] = filename.replace('.npy', '')

        test_df.append(image_knn_data)
    
    test_df = pd.concat(test_df).reset_index(drop=True)
    test_df['confidence'] = 1 - pd.to_numeric(test_df['distances'])
    test_df = test_df.groupby(['id', 'target']).confidence.max().reset_index()
    test_df = test_df.sort_values('confidence', ascending=False).reset_index(drop=True)
    # test_df['target'] = test_df['target'].map(decodeSparseId)

    predictions = {}

    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):

        target_id = decodeSparseId(id_encodings, row.target)

        if row.id in predictions:
            if len(predictions[row.id]) == 5:
                continue
            predictions[row.id].append(target_id)

        elif row.confidence > new_i_threshold:
            predictions[row.id] = [target_id, 'new_individual']

        else:
            predictions[row.id] = ['new_individual', target_id]
            
    for i in tqdm(predictions):

        if len(predictions[i]) < 5:
            remaining = [y for y in sample_list if y not in predictions]
            predictions[i] = predictions[i] + remaining
            predictions[i] = predictions[i][:5]

        predictions[i] = ' '.join(predictions[i])
        
    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ['image', 'predictions']

    predictions.to_csv('projects/happywhale-2022/data/metadata/submissions/knn_b7_submission.csv', index=False)


def b7():

    data_dir = 'projects/happywhale-2022/data/data_numpy_768_idxs/'
    filenames = os.listdir(data_dir)
    embeds_save_dir = 'projects/happywhale-2022/data/data_embeddings_768/b7/'

    for fold in tqdm(FOLDS):

        if fold == '0':

            filenames = filenames[22096:]
        
        else:

            filenames = os.listdir(data_dir)

        model_arc = createArcModel()
        weights_path = 'projects/happywhale-2022/models/effnetb7_loaded/effnetv1_b7_loss_' + fold + '.h5'
        model_arc.load_weights(weights_path)

        model_embed = tf.keras.models.Model(inputs=[model_arc.inputs[0]], outputs=[model_arc.layers[-4].output])

        del model_arc

        for filename in tqdm(filenames):

            image_path = data_dir + filename
            image = prepareImageB7(image_path)
            embeddings = model_embed(image)[0].numpy()
            # embeddings = getEmbeddings(image_path, embed_model, model_name)
            embeddings_save_path = embeds_save_dir + fold + '/' + filename.replace('.npy', '') + '_' + fold + '.npy'
            saveNumpy(embeddings, embeddings_save_path)


def b5():

    data_dir = 'projects/happywhale-2022/data/data_numpy_768_idxs/'
    filenames = os.listdir(data_dir)
    embeds_save_dir = 'projects/happywhale-2022/data/data_embeddings_768/b5/'

    for fold in tqdm(FOLDS):

        if fold == '4':
            break

        model_arc = load_model('projects/happywhale-2022/models/ids_arcmargin_sparse_kfold/' + fold + '/')

        model_embed = tf.keras.models.Model(inputs=[model_arc.inputs[0]], outputs=[model_arc.layers[-4].output])

        for filename in tqdm(filenames):

            image_path = data_dir + filename
            embeddings_save_path = embeds_save_dir + fold + '/' + filename.replace('.npy', '') + '_' + fold + '.npy'

            image = prepareImageB5(image_path, 'EfficientNetB5')
            embeddings = model_embed(image)[0].numpy()
            # embeddings = getEmbeddings(image_path, embed_model, model_name)
            saveNumpy(embeddings, embeddings_save_path)


def finalPrediction():

    new_i_thresholds = [0.2, 0.3, 0.4, 0.5]
    knn_neighbors = [100, 200, 500, 1000]
    sample_list = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00','956562ff2888']

    id_encodings = pd.read_csv('projects/happywhale-2022/data/metadata/individual_ids_idxs.csv')

    train_embeds_dir = 'projects/happywhale-2022/data/data_embeddings_768/b7/mean/'
    train_embeds_filenames = os.listdir(train_embeds_dir)
    train_embeds = [loadNumpy(train_embeds_dir + filename) for filename in train_embeds_filenames]
    train_embeds = np.asarray(train_embeds)

    train_targets = [filename.split('_')[-1].replace('.npy', '') for filename in train_embeds_filenames]
    train_targets = np.asarray(train_targets)

    test_embeds_dir = 'projects/happywhale-2022/data/test_embeddings_768/b7/mean/'
    test_embeds_filenames = os.listdir(test_embeds_dir)
    test_embeds = [loadNumpy(test_embeds_dir + filename) for filename in test_embeds_filenames]
    test_embeds = np.asarray(test_embeds)

    for new_i_threshold in tqdm(new_i_thresholds):

        for knn_neighbor in tqdm(knn_neighbors):

            knn = NearestNeighbors(n_neighbors=knn_neighbor, metric='cosine')
            knn.fit(train_embeds)

            distances, idxs = knn.kneighbors(test_embeds, knn_neighbor, return_distance=True)
            targets = train_targets[idxs]

            test_df = []
            
            for idx, image_id in enumerate(test_embeds_filenames):
                knn_data = [targets[idx], distances[idx]]
                knn_data = np.stack(knn_data, axis=1)
                image_knn_data = pd.DataFrame(knn_data, columns=['target', 'distances'])
                image_knn_data['image'] = image_id.replace('.npy', '')
                test_df.append(image_knn_data)
            
            test_df = pd.concat(test_df).reset_index(drop=True)
            test_df['confidence'] = 1 - pd.to_numeric(test_df['distances'])
            test_df = test_df.groupby(['image', 'target']).confidence.max().reset_index()
            test_df = test_df.sort_values('confidence', ascending=False).reset_index(drop=True)

            predictions = {}

            for i, row in test_df.iterrows():

                target_id = decodeSparseId(id_encodings, row.target)

                if row.image in predictions:
                    if len(predictions[row.image]) == 5:
                        continue
                    elif row.confidence > new_i_threshold:
                        predictions[row.image].append(target_id)
                    else:
                        predictions[row.image].append('new_individual')

                elif row.confidence > new_i_threshold:
                    predictions[row.image] = [target_id]

                else:
                    predictions[row.image] = ['new_individual']
                    
            for i in predictions:
                if len(predictions[i]) < 5:
                    print('len is less than 5')
                    remaining = [y for y in sample_list if y not in predictions]
                    predictions[i] = predictions[i] + remaining
                    predictions[i] = predictions[i][:5]

                predictions[i] = ' '.join(predictions[i])
                
            predictions = pd.Series(predictions).reset_index()
            predictions.columns = ['image', 'predictions']

            predictions.to_csv('projects/happywhale-2022/data/metadata/submissions/knn_b7_' + str(new_i_threshold) + '_' + str(knn_neighbor) + '_submission.csv', index=False)
            
