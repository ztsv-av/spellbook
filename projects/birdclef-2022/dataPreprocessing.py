import numpy as np
import pandas as pd
import time
import json
import os
import librosa
import librosa.display
import shutil
import matplotlib.pyplot as plt
import sounddevice as sd
import random
from tqdm.notebook import tqdm

os.chdir('V:\Git\spellbook')

from globalVariables import INDENT_SECONDS, SAMPLING_RATE, SIGNAL_LENGTH, HOP_LENGTH, N_FFT, N_MELS, WIN_LENGTH, FMIN
from helpers import saveNumpy, loadNumpy, evaluateString


def createSparseMeta(data_dir, save_meta_path):

    labels = os.listdir(data_dir)
    sparse_idx = 0
    sparse_list = []
    for label in labels:
        sparse_list.append([label, sparse_idx])
        sparse_idx += 1
    
    sparse_meta = pd.DataFrame(data=sparse_list, columns=['label', 'idx'])
    sparse_meta.to_csv(path_or_buf=save_meta_path, index=False)


def prepareFilenames(data_dir, sparse_meta, save_dir):

    labels = os.listdir(data_dir)
    for label in tqdm(labels):

        label_dir = data_dir + label + '/'
        filenames = os.listdir(label_dir)

        for filename in filenames:

            sparse_idx = sparse_meta[sparse_meta['label'] == label]['idx'].values[0]

            file_path = label_dir + filename
            c_file_path = save_dir + filename.replace('.ogg', '') + '_' + sparse_idx + '.ogg'
            shutil.copy(file_path, c_file_path)


def sortByRating(data_dir, data_meta, sparse_meta, save_dir):

    filenames = os.listdir(data_dir)

    for filename in tqdm(filenames):

        sparse_idx = filename.split('_')[-1].replace('.ogg', '')
        label = sparse_meta[sparse_meta['idx'] == evaluateString(sparse_idx)]['label'].values[0]

        filename_m = label + '/' + filename.split('_')[0] + '.ogg'

        rating = data_meta[data_meta['filename'] == filename_m]['rating'].values[0]
        if rating <= 2:
            continue

        file_path = data_dir + filename
        c_file_path = save_dir + filename
        shutil.copy(file_path, c_file_path)


def plotAudio(data, file_path, sr, threshold, ogg):

    if file_path is None:
        audio = data
    elif ogg:
        audio, sr = librosa.load(file_path, sr=sr)
    else:
        audio = loadNumpy(file_path)

    seconds = audio.shape[0] / sr
    print('Length: ', seconds)

    if threshold is None:

        plt.figure()
        plt.plot(audio)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.plot()

    else:

        fig, ax = plt.subplots()
        ax.plot(audio)
        ax.hlines(y=threshold, xmin=0, xmax=audio.shape[0], color='r')
        plt.show()


def plotOgg(file_path, sr):

    audio, sr = librosa.load(file_path, sr=sr)
    librosa.display.waveplot(audio, sr=sr)


def playNumpy(data, file_path, sr):

    if file_path is not None:
        data = loadNumpy(file_path)
    sd.play(data, sr)


def removeNoise(data_dir, sr, save_dir):

    data_filenames = os.listdir(data_dir)
    for filename in tqdm(data_filenames):

        file_path = data_dir + filename

        sig, rate = librosa.load(file_path, sr=sr, mono=True, res_type="kaiser_fast")

        mean = np.mean(np.abs(sig))
        sig_max = np.max(sig)
        threshold = (sig_max + 4.5*mean)/3.5

        idxs = np.ravel(np.argwhere(sig >= threshold))

        if len(idxs) == 0:
            continue

        indent_15_sec = int(INDENT_SECONDS * 1.5)
        total_idxs=[range(x - indent_15_sec, x + indent_15_sec + 1, 1) for x in idxs]

        new_range = [total_idxs[0][0], total_idxs[0][-1]]
        new_ranges = []
        for index, v in enumerate(total_idxs[1:]):

            v_min = v[0]
            v_max = v[-1]
            v_prev_max = total_idxs[index][-1]

            if v_prev_max in v:
                new_range[1] = v_max
            
            else:
                new_ranges.append(new_range)
                new_range = [v_min, v_max]
        new_ranges.append(new_range)
        
        sigs_trimmed = []
        for new_range in new_ranges:
            min_idx = new_range[0]
            if min_idx < 0:
                min_idx = 0
            max_idx = new_range[1]
            if max_idx > len(sig):
                max_idx = len(sig) - 1
            trim_idxs = range(min_idx, max_idx, 1)
            sigs_trimmed.append(sig[trim_idxs])
        
        sig_trimmed = np.concatenate(sigs_trimmed, axis=0)
        
        save_path = save_dir + filename.replace('ogg', 'npy')
        saveNumpy(sig_trimmed, save_path)


def trim5Sec(data_dir, sr, save_dir, ogg):

    data_filenames = os.listdir(data_dir)
    for filename in tqdm(data_filenames):

        file_path = data_dir + filename
        if ogg:
            audio, _ = librosa.load(file_path, sr=sr)
        else:
            audio = loadNumpy(file_path)

        step_5_sec = SIGNAL_LENGTH * INDENT_SECONDS - 1
        step_35_sec = 3.5 * INDENT_SECONDS - 1
        counter = SIGNAL_LENGTH

        if len(audio) < step_5_sec:

            split = np.zeros(step_5_sec)
            split[:len(audio)] += audio

            save_path = save_dir + filename.replace('.npy', '').replace('.ogg', '') + '_' + str(counter) + '.npy'
            saveNumpy(split, save_path)

        else:

            for i in range(0, len(audio), step_5_sec):

                split = audio[i:i + step_5_sec]

                if len(split) < step_5_sec:

                    if len(split) < step_35_sec:
                        continue
                    else:
                        split_pad = np.zeros(step_5_sec)
                        split_pad[:len(split)] += split

                    save_path = save_dir + filename.replace('.npy', '').replace('.ogg', '') + '_' + str(counter) + '.npy'
                    saveNumpy(split_pad, save_path)
                
                else:

                    save_path = save_dir + filename.replace('.npy', '').replace('.ogg', '') + '_' + str(counter) + '.npy'
                    saveNumpy(split, save_path)

                counter += SIGNAL_LENGTH


def createMelspecs(data_dir, save_dir):

    filenames = os.listdir(data_dir)
    for filename in tqdm(filenames):

        file_path = data_dir + filename
        audio = loadNumpy(file_path)

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLING_RATE, win_length=WIN_LENGTH, 
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN)

        save_path = save_dir + filename
        saveNumpy(mel_spec, save_path)


def oldToNewIdxs():

    with open('projects/birdclef-2022/data/metadata/scored_birds.json') as sbfile:
        scored_birds_meta = json.load(sbfile)
    sparse_meta = pd.read_csv('projects/birdclef-2022/data/metadata/sparse_metadata.csv')
    birds_sparse_idxs = [sparse_meta[sparse_meta['label'] == bird]['idx'].values[0] for bird in scored_birds_meta]
    birds_idxs = {bird: bird_idx for bird, bird_idx in zip(scored_birds_meta, birds_sparse_idxs)}

    new_idxs = [i for i in range(21)]
    new_idxs_to_old = []
    for old, new in zip(birds_idxs.values(), new_idxs):
        new_idxs_to_old.append([old, new])

    new_old_meta = pd.DataFrame(new_idxs_to_old, columns=['old_idx', 'new_idx'])
    new_old_meta.to_csv(path_or_buf='projects/birdclef-2022/data/metadata/old_to_new_sparse.csv', index=False)


def recreate21Classes():

    with open('projects/birdclef-2022/data/metadata/scored_birds.json') as sbfile:
        scored_birds_meta = json.load(sbfile)
    sparse_meta = pd.read_csv('projects/birdclef-2022/data/metadata/sparse_metadata.csv')
    birds_sparse_idxs = [sparse_meta[sparse_meta['label'] == bird]['idx'].values[0] for bird in scored_birds_meta]
    birds_idxs = {bird: str(bird_idx) for bird, bird_idx in zip(scored_birds_meta, birds_sparse_idxs)}
    new_old_idxs = pd.read_csv('projects/birdclef-2022/data/metadata/old_to_new_sparse.csv')

    data_dir = 'projects/birdclef-2022/data/data_melspecs/'
    data_save_dir = 'projects/birdclef-2022/data/data_melspecs_22_less30/'

    for file in tqdm(os.listdir(data_dir)):

        file_path = data_dir + file

        bird_idx = file.split('_')[1]
        end_time = file.split('_')[-1].replace('.npy', '')

        if bird_idx not in birds_idxs.values():
            if int(end_time) > 30:
                continue
            bird_idx = '21'
            new_file_path = data_save_dir + file.split('_')[0] + '_' + bird_idx + '_' + file.split('_')[-1]
            shutil.copy(file_path, new_file_path)
        else:
            bird_idx = str(new_old_idxs[new_old_idxs['old_idx'] == evaluateString(bird_idx)]['new_idx'].values[0])
            new_file_path = data_save_dir + file.split('_')[0] + '_' + bird_idx + '_' + file.split('_')[-1]
            shutil.copy(file_path, new_file_path)
