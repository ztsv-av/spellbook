import tensorflow as tf

import os
import pandas as pd
import numpy as np
import time
import shutil
from tqdm.notebook import tqdm

os.chdir('V:\Git\spellbook')

from globalVariables import PERMUTATIONS_CLASSIFICATION
from permutationFunctions import classification_permutations
from preprocessFunctions import minMaxNormalizeNumpy, addColorChannels
from helpers import loadImage, saveNumpyArray, visualizeImageBox, loadNumpy, splitTrainValidation, evaluateString


def preprocessData():

    files_dir = 'projects/happywhale-2022/data/data_images/'
    files = os.listdir(files_dir)

    save_dir = 'projects/happywhale-2022/data/data_numpy_768/'

    reshape_size = (768, 768)

    other_images = []

    for filename in tqdm(files, desc='Images Processed'):

        path = files_dir + filename

        image = loadImage(path, image_type='uint8')

        if len(image.shape) == 2:

            image = addColorChannels(image, 3)
        
        elif len(image.shape) > 3:

            other_images.append(filename)

            continue

        image = tf.image.resize(image, reshape_size)
        image = tf.cast(image, tf.uint8).numpy()

        saveNumpyArray(image, save_dir + filename)


def metaNamesReplacement():

    df = pd.read_csv('projects/happywhale-2022/data/metadata/train_metadata.csv')

    df.species.replace({"globis": "short_finned_pilot_whale",
                            "pilot_whale": "short_finned_pilot_whale",
                            "kiler_whale": "killer_whale",
                            "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)
    
    df.to_csv(path_or_buf='projects/happywhale-2022/data/metadata/train_preprocessed_metadata.csv', index=False)


def oneHotSpeciesIds():

    df = pd.read_csv('projects/happywhale-2022/data/metadata/train_preprocessed_metadata.csv')

    ids = df['id'].values
    species = df['species'].values
    individual_ids = df['individual_id'].values

    unique_species, indicies_species = np.unique(species, return_inverse=True)
    unique_individual_ids, indicies_individual_ids = np.unique(individual_ids, return_inverse=True)

    onehot_species = np.eye(unique_species.shape[0])[indicies_species].astype(int)
    onehot_individual_ids = np.eye(unique_individual_ids.shape[0])[indicies_individual_ids].astype(int)

    one_hot_metadata_list = []

    for idx, value in enumerate(ids):

        one_hot_metadata_list.append([value, onehot_species[idx].tolist(), onehot_individual_ids[idx].tolist()])

    one_hot_metadata = pd.DataFrame(one_hot_metadata_list, columns=['id', 'species', 'individual_id'])
    one_hot_metadata.to_csv(path_or_buf='projects/happywhale-2022/data/metadata/train_onehot_metadata.csv', index=False)


def flipImages():

    files_dir = 'projects/happywhale-2022/data/data_numpy_384/'
    files = os.listdir(files_dir)

    save_dir = 'projects/happywhale-2022/data/data_numpy_384_flipped/'

    for filename in files:

        path = files_dir + filename

        data = loadNumpy(path)
        data_f = classification_permutations(data, PERMUTATIONS_CLASSIFICATION)

        saveNumpyArray(data, save_dir + filename.replace('.npy', '') + '_nf')
        saveNumpyArray(data_f, save_dir + filename.replace('.npy', '') + '_f')


def checkPermutations():

    files_dir = 'projects/happywhale-2022/data/data_numpy_384_flipped/'
    files = os.listdir(files_dir)

    num_files = 15

    for idx, filename in enumerate(files):

        path = files_dir + filename

        data = loadNumpy(path)
        data_f = classification_permutations(data, PERMUTATIONS_CLASSIFICATION)

        # use matplotlib inline
        visualizeImageBox(data, None)
        visualizeImageBox(data_f, None)

        if idx == num_files:
            break


def renameWithSpecies():

    files_dir = 'projects/happywhale-2022/data/data_numpy_768/'
    new_files_dir = 'projects/happywhale-2022/data/data_numpy_768_idxs/'

    df = pd.read_csv('projects/happywhale-2022/data/metadata/train_preprocessed_metadata.csv')

    flips = False

    ids = df['id'].values
    species = df['species'].values
    individual_ids = df['individual_id'].values

    unique_species, indicies_species = np.unique(species, return_inverse=True)
    unique_individual_ids, indicies_individual_ids = np.unique(individual_ids, return_inverse=True)

    onehot_species = np.eye(unique_species.shape[0])[indicies_species].astype(int)
    onehot_individual_ids = np.eye(unique_individual_ids.shape[0])[indicies_individual_ids].astype(int)

    for idx, file in tqdm(enumerate(ids), desc='Files Processed'):

        if flips:

            old_path_no_flip = files_dir + file

            flips = ['_f', '_nf']

            for flip in flips:

                old_path = old_path_no_flip + flip + '.npy'

                specie_idx =  np.argmax(onehot_species[idx]) + 1
                individual_idx = np.argmax(onehot_individual_ids[idx]) + 1

                new_path = new_files_dir + file.replace('.jpg', '') + '_' + str(specie_idx) + '_' + str(individual_idx) + flip + '.npy'

                shutil.copy(old_path, new_path)
        
        else:

            old_path = files_dir + file + '.npy'

            specie_idx =  np.argmax(onehot_species[idx]) + 1
            individual_idx = np.argmax(onehot_individual_ids[idx]) + 1

            new_path = new_files_dir + file.replace('.jpg', '') + '_' + str(specie_idx) + '_' + str(individual_idx) + '.npy'

            shutil.copy(old_path, new_path)


def createIdxSpecieIdMeta():

    df = pd.read_csv('projects/happywhale-2022/data/metadata/train_preprocessed_metadata.csv')

    ids = df['id'].values
    species = df['species'].values
    individual_ids = df['individual_id'].values

    unique_species, indicies_species = np.unique(species, return_inverse=True)
    unique_individual_ids, indicies_individual_ids = np.unique(individual_ids, return_inverse=True)

    species_list = []
    ids_list = []

    for idx, specie in enumerate(unique_species):

        species_list.append([specie, idx])
    
    for idx, i_id in enumerate(unique_individual_ids):

        ids_list.append([i_id, idx])
    
    species_metadata = pd.DataFrame(species_list, columns=['specie', 'idx'])
    species_metadata.to_csv(path_or_buf='projects/happywhale-2022/data/metadata/species_idxs.csv', index=False)

    ids_metadata = pd.DataFrame(ids_list, columns=['individual_id', 'idx'])
    ids_metadata.to_csv(path_or_buf='projects/happywhale-2022/data/metadata/individual_ids_idxs.csv', index=False)
