import pandas as pd
import numpy as np
import os
import random
import librosa
import librosa.display
import time
import shutil
import itertools
from pydub import AudioSegment
from scipy.io import wavfile
from sklearn.utils import shuffle
from PIL import Image
from pydub import AudioSegment
from sklearn.model_selection import train_test_split

from globalVariables import (DATA_FILEPATHS, INDENT_SECONDS, METADATA, INPUT_SHAPE, RANDOM_STATE, SAMPLING_RATE, SIGNAL_LENGTH, N_FFT, FMIN, FMAX)


PREPROCESSED_SAVE_DIR_NOISE = ''
PREPROCESSED_SAVE_DIR_WAV = ''
PREPROCESSED_SAVE_DIR_MS = ''
PREPROCESSED_SAVE_DIR = ''
FOREST_PATH = ''
RAIN_PATH = ''
FOREST_AUDIO = AudioSegment.from_file(FOREST_PATH, format="wav")
RAIN_AUDIO = AudioSegment.from_file(RAIN_PATH, format="wav")

METADATA_RATED = METADATA[METADATA['rating'] >= 3.5]
FILES = list(METADATA['filename'])
LABELS = METADATA['primary_label'].unique()
CLASS_TO_INT = {k: v for v, k in enumerate(LABELS)}
idxs = list(CLASS_TO_INT.values())

WAV_DIR_TWO_CLASSES = 'datasets/train/wav_trimmed_train_short_3.5_2classes/'
WAV_MERGED_DIR_TWO_CLASSES = 'datasets/train/wav_merged_trimmed_train_short_3.5_2classes/'
PREPROCESSED_SAVE_DIR_TWO_CLASSES = 'datasets/train/trimmed_merged_train_short_3.5_2classes/'
WAV_DIR_THREE_CLASSES = 'datasets/train/wav_trimmed_train_short_3.5_3classes/'
WAV_MERGED_DIR_THREE_CLASSES = 'datasets/train/wav_merged_trimmed_train_short_3.5_3classes/'
PREPROCESSED_SAVE_DIR_THREE_CLASSES = 'datasets/train/trimmed_merged_train_short_3.5_3classes/'


def forestOverlay(audio_path, audio_name, dest_repo, numpy_dir):

    audio = AudioSegment.from_file(audio_path, format="wav")

    random_idx = random.randint(0, int(len(FOREST_AUDIO) * 0.9))
    noise = FOREST_AUDIO[random_idx:random_idx + len(audio)]

    # forest 21 dB louder
    louder_noise = noise + 21

    # Overlay audio over noise at position 0
    overlay = audio.overlay(louder_noise, position=0)

    # simple export
    if not os.path.exists(dest_repo):
        os.makedirs(dest_repo)
    dest_path = dest_repo + 'forest_' + audio_name + '.wav'
    file_export_wav = overlay.export(dest_path, format="wav")

    sig_int, rate = librosa.load(dest_path, sr=32000, offset=None)
    sig_16 = (sig_int* 32767).astype(int)
    sig_float = sig_16/32767.0

    numpy_path = numpy_dir + 'forest_' + audio_name

    np.save(numpy_path, sig_float)
    

def rainOverlay(audio_path, audio_name, dest_repo, numpy_dir):

    audio = AudioSegment.from_file(audio_path, format="wav")

    random_idx = random.randint(0, int(len(RAIN_AUDIO) * 0.9))
    noise = RAIN_AUDIO[random_idx:random_idx + len(audio)]

    # rain 8 dB louder
    rain_louder = noise + 8

    # Overlay audio over noise at position 0
    overlay = audio.overlay(rain_louder, position=0)

    # simple export
    if not os.path.exists(dest_repo):
        os.makedirs(dest_repo)
    dest_path = dest_repo + 'rain_' + audio_name + '.wav'
    file_export_wav = overlay.export(dest_path, format="wav")

    sig_int, rate = librosa.load(dest_path, sr=32000, offset=None)
    sig_16 = (sig_int* 32767).astype(int)
    sig_float = sig_16/32767.0

    numpy_path = numpy_dir + 'rain_' + audio_name

    np.save(numpy_path, sig_float)


def rainForestOverlay(audio_path, audio_name, dest_repo, numpy_dir):

    audio = AudioSegment.from_file(audio_path, format="wav")

    # rain idx
    rain_random_idx = random.randint(0, int(len(RAIN_AUDIO) * 0.9))
    rain_noise = RAIN_AUDIO[rain_random_idx:rain_random_idx + len(audio)]

    # forest idx
    forest_random_idx = random.randint(0, int(len(FOREST_AUDIO) * 0.9))
    forest_noise = FOREST_AUDIO[forest_random_idx:forest_random_idx + len(audio)]

    # rain 8 dB louder
    rain_louder = rain_noise + 8

    # forest 21 dB louder
    forest_louder = forest_noise + 21

    # rain + forest overlay over audio
    rain_overlay = audio.overlay(rain_louder, position=0)
    forest_rain_overlay = rain_overlay.overlay(forest_louder, position=0)

    # simple export
    if not os.path.exists(dest_repo):
        os.makedirs(dest_repo)
    dest_path = dest_repo + 'rain_forest_' + audio_name + '.wav'
    file_export_wav = forest_rain_overlay.export(dest_path, format="wav")

    sig_int, rate = librosa.load(dest_path, sr=32000, offset=None)
    sig_16 = (sig_int* 32767).astype(int)
    sig_float = sig_16/32767.0

    numpy_path = numpy_dir + 'rain_forest_' + audio_name

    np.save(numpy_path, sig_float)


def addNoiseOverlay():

    root, labels, files = os.walk(DATA_FILEPATHS).__next__()

    for label in labels:

        print('STARTING', str(label), '!')

        data_repo = DATA_FILEPATHS + str(label) + '/'
        dest_repo = PREPROCESSED_SAVE_DIR_NOISE + str(label) + '/'

        if not os.path.exists(dest_repo):
            os.makedirs(dest_repo)

        numpy_repo = 'datasets/train/noisy_numpy_train_full/' + str(label) + '/'
        if not os.path.exists(numpy_repo):
            os.makedirs(numpy_repo)

        audios = os.listdir(data_repo)

        n_elements = int(0.4 * len(audios)) # 40% of noise augmentation
        if len(audios) <= 2 :   # check if 40% is less than 3 files
            n_elements = 1
        elif len(audios) <= 4:
            n_elements = 2
        
        random_files_to_noise = random.sample(audios, n_elements)   # # 40% of files from 1 class to overlay with noise
        
        if len(random_files_to_noise) <= 2: # if 40% is less than 3 files => do forest_rain augmentation on those two or less files

            for random_file in random_files_to_noise:

                numpy_path = data_repo + random_file
                numpy_audio = np.load(numpy_path)
                numpy_audio = np.int16(numpy_audio/np.max(np.abs(numpy_audio)) * 32767)

                wav_name = str(random_file.replace('.npy', ''))
                wav_path = dest_repo + wav_name + '.wav'
                wavfile.write(wav_path, 32000, numpy_audio) # write numpy to wav ('filename.wav')
                    
                rainForestOverlay(wav_path, wav_name, dest_repo, numpy_repo)
                
                os.remove(numpy_path)   # remove original numpy file
            
            print('  FINISHED WITH', str(label), '!')
            
            continue

        forest_len = int(len(audios) * 0.15) # 15% from num_of_audios to forest aug
        rain_len = int(len(audios) * 0.15) # 15% from num_of_audios to rain aug

        # forest augmentation

        forest_time_start = time.time()

        for random_file in random_files_to_noise[:forest_len]:
            
            numpy_path = data_repo + random_file
            numpy_audio = np.load(numpy_path)
            numpy_audio = np.int16(numpy_audio/np.max(np.abs(numpy_audio)) * 32767)

            wav_name = str(random_file.replace('.npy', ''))
            wav_path = dest_repo + wav_name + '.wav'
            wavfile.write(wav_path, 32000, numpy_audio) # write numpy to wav ('filename.wav')

            numpy_dir = numpy_repo + wav_name

            forestOverlay(wav_path, wav_name, dest_repo, numpy_repo)
            
            os.remove(numpy_path)   # remove original numpy file

        forest_time_end = time.time()
        print(' TIME SPEND FOR FOREST AUGMENTATION', round(forest_time_end - forest_time_start, 2))

        # rain augmentation

        rain_time_start = time.time()

        for random_file in random_files_to_noise[forest_len:forest_len + rain_len]:
            
            numpy_path = data_repo + random_file
            numpy_audio = np.load(numpy_path)
            numpy_audio = np.int16(numpy_audio/np.max(np.abs(numpy_audio)) * 32767)

            wav_name = str(random_file.replace('.npy', ''))
            wav_path = dest_repo + wav_name + '.wav'
            wavfile.write(wav_path, 32000, numpy_audio) # write numpy to wav ('filename.wav')
            
            numpy_dir = numpy_repo + wav_name

            rainOverlay(wav_path, wav_name, dest_repo, numpy_repo)
            
            os.remove(numpy_path)   # remove original numpy file

        rain_time_end = time.time()
        print(' TIME SPEND FOR RAIN AUGMENTATION', round(rain_time_end - rain_time_start, 2))

        # forest + rain augmentation

        forest_rain_time_start = time.time()

        for random_file in random_files_to_noise[forest_len + rain_len:len(audios)]:
            
            numpy_path = data_repo + random_file
            numpy_audio = np.load(numpy_path)
            numpy_audio = np.int16(numpy_audio/np.max(np.abs(numpy_audio)) * 32767)

            wav_name = str(random_file.replace('.npy', ''))
            wav_path = dest_repo + wav_name + '.wav'
            wavfile.write(wav_path, 32000, numpy_audio) # write numpy to wav ('filename.wav')

            rainForestOverlay(wav_path, wav_name, dest_repo, numpy_repo)
            
            os.remove(numpy_path)   # remove original numpy file

        forest_rain_time_end = time.time()
        print(' TIME SPEND FOR FOREST + RAIN AUGMENTATION', round(forest_rain_time_end - forest_rain_time_start, 2))

        print('  FINISHED WITH', str(label), '!')


def mergeAudio_Melspec():

    print('PROCESSING MERGING, SPLITTING AND CREATING MELSPECTROGRAMS...\n')
    start_operation = time.time()

    root, labels, files = os.walk(DATA_FILEPATHS).__next__()

    for label in labels:

        # Concantenate audios of one class into one big audio
        
        full_audio = np.array([])

        repo = DATA_FILEPATHS + label
        audios = os.listdir(repo)

        print('  PROCESSING CONCATENATING', label, '...')
        start_concat_label = time.time()
        
        for audio in audios:

            file_path = repo + '/' + audio

            file = np.load(file_path)

            full_audio = np.concatenate((full_audio, file), axis=None)
        
        end_concat_label = time.time()

        print('   FINISHED CONCATENATING', label, '!')
        print('   TIME ELAPSED FOR CONCATENATING', label, round(end_concat_label - start_concat_label, 2), '\n')

        # Split signal into five second chunks
        
        print('   STARTING SPLITTING, SAVING SPLITS, CREATING MELSPECS AND SAVING MELSPECS OF', label, '!')
        start_melspec = time.time()
        
        file_count = 0

        for i in range(0, len(full_audio), int(5 * 32000)):

            split = full_audio[i:i + int(5 * 32000)]

            # End of signal?
            if len(split) < int(5 * 32000):
                break

            # Create melspectogram of 5 second split

            hop_length = int(SIGNAL_LENGTH * SAMPLING_RATE / (INPUT_SHAPE[1] - 1))
            mel_spec = librosa.feature.melspectrogram(y=split, 
                                                    sr=SAMPLING_RATE, 
                                                    n_fft=N_FFT, 
                                                    hop_length=hop_length, 
                                                    n_mels=INPUT_SHAPE[0], 
                                                    fmin=FMIN, 
                                                    fmax=FMAX)

            # Save melspectogram
            melspec_save_dir = PREPROCESSED_SAVE_DIR_MS + label
            if not os.path.exists(melspec_save_dir):
                os.makedirs(melspec_save_dir)
            save_path = (melspec_save_dir + '/' + label + '_' + str(file_count))
            np.save(save_path, mel_spec)

            file_count += 1

        end_melspec = time.time()
        print('   FINISHED WITH MELSPECTOGRAMS OF', label, '!')
        print('   TIME ELAPSED FOR MELSPECTOGRAMS OF', label, round(end_melspec - start_melspec, 2), '\n')

    end_operation = time.time()
    print('   FINISHED WITH OPERATION!')
    print('   TIME ELAPSED:', round(end_operation - start_operation, 2))


def classMerge(two_class, three_class):

    two_classes_len = len(FILES) * 0.3
    three_classes_len = len(FILES)

    if two_class:

        print('STARTING 2 CLASS MERGE...')
        two_class_start = time.time()

        two_classes_counter = 0

        while two_classes_counter < two_classes_len:
            
            first_class_int = int(np.random.uniform(low=0, high=396))
            second_class_int = int(np.random.uniform(low=0, high=396))
            if first_class_int == second_class_int:
                continue
            
            first_class = list(CLASS_TO_INT.keys())[list(CLASS_TO_INT.values()).index(first_class_int)]
            second_class = list(CLASS_TO_INT.keys())[list(CLASS_TO_INT.values()).index(second_class_int)]
            
            first_class_files = os.listdir(DATA_FILEPATHS + first_class + '/')
            second_class_files = os.listdir(DATA_FILEPATHS + second_class + '/')
            
            first_class_random_filename = np.random.choice(first_class_files)
            second_class_random_filename = np.random.choice(second_class_files)
            
            if (os.path.exists(PREPROCESSED_SAVE_DIR_TWO_CLASSES + first_class + '_' + second_class + '/' + first_class + '_' + second_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename) or 
                os.path.exists(PREPROCESSED_SAVE_DIR_TWO_CLASSES + second_class + '_' + first_class + '/' + second_class + '_' + first_class + '_' + second_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename)):
                continue
            
            print('     MERGING', first_class + '/' + first_class_random_filename, second_class + '/' + second_class_random_filename)
            
            first_class_numpy_file = np.load(DATA_FILEPATHS + first_class + '/' + first_class_random_filename)
            first_class_numpy = np.int16(first_class_numpy_file/np.max(np.abs(first_class_numpy_file)) * 32767)
            second_class_numpy_file = np.load(DATA_FILEPATHS + second_class + '/' + second_class_random_filename)
            second_class_numpy = np.int16(second_class_numpy_file/np.max(np.abs(second_class_numpy_file)) * 32767)
            
            first_class_wav_path = WAV_DIR_TWO_CLASSES + first_class + '/' + first_class_random_filename.replace('npy', 'wav')
            if not os.path.exists(WAV_DIR_TWO_CLASSES + first_class + '/'):
                os.makedirs(WAV_DIR_TWO_CLASSES + first_class + '/')
            second_class_wav_path = WAV_DIR_TWO_CLASSES + second_class + '/' + second_class_random_filename.replace('npy', 'wav')
            if not os.path.exists(WAV_DIR_TWO_CLASSES + second_class + '/'):
                os.makedirs(WAV_DIR_TWO_CLASSES + second_class + '/')
            
            wavfile.write(first_class_wav_path, 32000, first_class_numpy)
            wavfile.write(second_class_wav_path, 32000, second_class_numpy)
            
            first_class_wav = AudioSegment.from_file(first_class_wav_path, format="wav")
            second_class_wav = AudioSegment.from_file(second_class_wav_path, format="wav")
            
            if len(first_class_numpy) <= len(second_class_numpy):
                overlay = first_class_wav.overlay(second_class_wav, position=0)
            else:
                overlay = second_class_wav.overlay(first_class_wav, position=0)
            
            wav_merged_path_2_classes = (WAV_MERGED_DIR_TWO_CLASSES + first_class + '_' + second_class + '/' + 
                                        first_class + '_' + second_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('npy', 'wav'))
            
            if not os.path.exists(WAV_MERGED_DIR_TWO_CLASSES + first_class + '_' + second_class + '/'):
                os.makedirs(WAV_MERGED_DIR_TWO_CLASSES + first_class + '_' + second_class + '/')
            
            overlay.export(wav_merged_path_2_classes, format="wav")
            
            sig_int, rate = librosa.load(wav_merged_path_2_classes, sr=32000, offset=None)
            sig_16 = (sig_int* 32767).astype(int)
            sig_float = sig_16/32767.0

            numpy_merged_path_2_classes = (PREPROCESSED_SAVE_DIR_TWO_CLASSES + first_class + '_' + second_class + '/' + 
                                          first_class + '_' + second_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename)
            if not os.path.exists(PREPROCESSED_SAVE_DIR_TWO_CLASSES + first_class + '_' + second_class + '/'):
                os.makedirs(PREPROCESSED_SAVE_DIR_TWO_CLASSES + first_class + '_' + second_class + '/')
            np.save(numpy_merged_path_2_classes, sig_float)
            
            print('     FINISHED MERGING TWO FILES!')
            
            two_classes_counter += 1

        two_class_end = time.time()
        print('FINISHED 2 CLASS MERGE. TIME SPENT:', round(two_class_end - two_class_start, 2))

    if three_class:

        print('STARTING 3 CLASS MERGE...')
        three_class_start = time.time()

        three_classes_counter = 0

        while three_classes_counter < three_classes_len:
            
            for first_class_int in idxs:
            
                second_class_int = int(np.random.uniform(low=0, high=396))
                third_class_int = int(np.random.uniform(low=0, high=396))
                if first_class_int == second_class_int or first_class_int == third_class_int or second_class_int == third_class_int:
                    continue
                
                first_class = list(CLASS_TO_INT.keys())[list(CLASS_TO_INT.values()).index(first_class_int)]
                second_class = list(CLASS_TO_INT.keys())[list(CLASS_TO_INT.values()).index(second_class_int)]
                third_class = list(CLASS_TO_INT.keys())[list(CLASS_TO_INT.values()).index(third_class_int)]
                
                first_class_files = os.listdir(DATA_FILEPATHS + first_class + '/')
                second_class_files = os.listdir(DATA_FILEPATHS + second_class + '/')
                third_class_files = os.listdir(DATA_FILEPATHS + third_class + '/')
                
                first_class_random_filename = np.random.choice(first_class_files)
                second_class_random_filename = np.random.choice(second_class_files)
                third_class_random_filename = np.random.choice(third_class_files)
                
                if (os.path.exists(PREPROCESSED_SAVE_DIR_THREE_CLASSES + first_class + '_' + second_class + '_' + third_class + '/' + first_class + '_' + second_class + '_' + third_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename) or
                    os.path.exists(PREPROCESSED_SAVE_DIR_THREE_CLASSES + first_class + '_' + third_class + '_' + second_class + '/' + first_class + '_' + third_class + '_' + second_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename) or 
                    os.path.exists(PREPROCESSED_SAVE_DIR_THREE_CLASSES + second_class + '_' + first_class + '_' + third_class + '/' + second_class + '_' + first_class + '_' + third_class + '_' + second_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename) or
                    os.path.exists(PREPROCESSED_SAVE_DIR_THREE_CLASSES + second_class + '_' + third_class + '_' + first_class + '/' + second_class + '_' + third_class + '_' + first_class + '_' + second_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename) or
                    os.path.exists(PREPROCESSED_SAVE_DIR_THREE_CLASSES + third_class + '_' + first_class + '_' + second_class + '/' + third_class + '_' + first_class + '_' + second_class + '_' + third_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename) or
                    os.path.exists(PREPROCESSED_SAVE_DIR_THREE_CLASSES + third_class + '_' + second_class + '_' + first_class + '/' + third_class + '_' + second_class + '_' + first_class + '_' + third_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename)):
                    continue
                
                print('     MERGING', first_class + '/' + first_class_random_filename, second_class + '/' + second_class_random_filename, third_class + '/' + third_class_random_filename)
                
                first_class_numpy_file = np.load(DATA_FILEPATHS + first_class + '/' + first_class_random_filename)
                first_class_numpy = np.int16(first_class_numpy_file/np.max(np.abs(first_class_numpy_file)) * 32767)
                second_class_numpy_file = np.load(DATA_FILEPATHS + second_class + '/' + second_class_random_filename)
                second_class_numpy = np.int16(second_class_numpy_file/np.max(np.abs(second_class_numpy_file)) * 32767)
                third_class_numpy_file = np.load(DATA_FILEPATHS + third_class + '/' + third_class_random_filename)
                third_class_numpy = np.int16(third_class_numpy_file/np.max(np.abs(third_class_numpy_file)) * 32767)
                
                first_class_wav_path = WAV_DIR_THREE_CLASSES + first_class + '/' + first_class_random_filename.replace('npy', 'wav')
                if not os.path.exists(WAV_DIR_THREE_CLASSES + first_class + '/'):
                    os.makedirs(WAV_DIR_THREE_CLASSES + first_class + '/')
                second_class_wav_path = WAV_DIR_THREE_CLASSES + second_class + '/' + second_class_random_filename.replace('npy', 'wav')
                if not os.path.exists(WAV_DIR_THREE_CLASSES + second_class + '/'):
                    os.makedirs(WAV_DIR_THREE_CLASSES + second_class + '/')
                third_class_wav_path = WAV_DIR_THREE_CLASSES + third_class + '/' + third_class_random_filename.replace('npy', 'wav')
                if not os.path.exists(WAV_DIR_THREE_CLASSES + third_class + '/'):
                    os.makedirs(WAV_DIR_THREE_CLASSES + third_class + '/')
                
                wavfile.write(first_class_wav_path, 32000, first_class_numpy)
                wavfile.write(second_class_wav_path, 32000, second_class_numpy)
                wavfile.write(third_class_wav_path, 32000, third_class_numpy)
                
                first_class_wav = AudioSegment.from_file(first_class_wav_path, format="wav")
                second_class_wav = AudioSegment.from_file(second_class_wav_path, format="wav")
                third_class_wav = AudioSegment.from_file(third_class_wav_path, format="wav")
                
                if len(first_class_numpy) <= len(second_class_numpy):
                    if len(first_class_numpy) <= len(third_class_numpy):
                        overlay1 = first_class_wav.overlay(second_class_wav, position=0)
                        overlay2 = overlay1.overlay(third_class_wav, position=0)
                    else:
                        overlay1 = third_class_wav.overlay(first_class_wav, position=0)
                        overlay2 = overlay1.overlay(second_class_wav, position=0)
                elif len(first_class_numpy) <= len(third_class_numpy):
                        overlay1 = second_class_wav.overlay(first_class_wav, position=0)
                        overlay2 = overlay1.overlay(third_class_wav, position=0)
                elif len(second_class_numpy) <= len(first_class_numpy):
                    if len(second_class_numpy) <= len(third_class_numpy):
                        overlay1 = second_class_wav.overlay(first_class_wav, position=0)
                        overlay2 = overlay1.overlay(third_class_wav, position=0)
                    else:
                        overlay1 = third_class_wav.overlay(second_class_wav, position=0)
                        overlay2 = overlay1.overlay(first_class_wav, position=0)
                elif len(first_class_numpy) <= len(second_class_numpy):
                        overlay1 = first_class_wav.overlay(second_class_wav, position=0)
                        overlay2 = overlay1.overlay(third_class_wav, position=0)
                elif len(third_class_numpy) <= len(first_class_numpy):
                    if len(third_class_numpy) <= len(second_class_numpy):
                        overlay1 = third_class_wav.overlay(first_class_wav, position=0)
                        overlay2 = overlay1.overlay(second_class_wav, position=0)
                    else:
                        overlay1 = second_class_wav.overlay(third_class_wav, position=0)
                        overlay2 = overlay1.overlay(first_class_wav, position=0)
                elif len(first_class_numpy) <= len(second_class_numpy):
                        overlay1 = first_class_wav.overlay(third_class_wav, position=0)
                        overlay2 = overlay1.overlay(second_class_wav, position=0)
                else:
                    print('         COULD NOT MERGE THREE FILES:', first_class + '/' + first_class_random_filename, second_class + '/' + second_class_random_filename, third_class + '/' + third_class_random_filename)
                    print('LEN FIRST FILE:', len(first_class_numpy), 'LEN SECOND FILE:', len(second_class_numpy), 'LEN THIRD FILE:', len(third_class_numpy))
                
                wav_merged_path_3_classes = WAV_MERGED_DIR_THREE_CLASSES + first_class + '_' + second_class + '_' + third_class + '/' + first_class + '_' + second_class + '_' + third_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename.replace('npy', 'wav')
                
                if not os.path.exists(WAV_MERGED_DIR_THREE_CLASSES + first_class + '_' + second_class + '_' + third_class + '/'):
                    os.makedirs(WAV_MERGED_DIR_THREE_CLASSES + first_class + '_' + second_class + '_' + third_class + '/')
                
                overlay2.export(wav_merged_path_3_classes, format="wav")
                
                sig_int, rate = librosa.load(wav_merged_path_3_classes, sr=32000, offset=None)
                sig_16 = (sig_int* 32767).astype(int)
                sig_float = sig_16/32767.0

                numpy_merged_path_3_classes = PREPROCESSED_SAVE_DIR_THREE_CLASSES + first_class + '_' + second_class + '_' + third_class + '/' + first_class + '_' + second_class + '_' + third_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename
                if not os.path.exists(PREPROCESSED_SAVE_DIR_THREE_CLASSES + first_class + '_' + second_class + '_' + third_class + '/'):
                    os.makedirs(PREPROCESSED_SAVE_DIR_THREE_CLASSES + first_class + '_' + second_class + '_' + third_class + '/')
                np.save(numpy_merged_path_3_classes, sig_float)
                
                print('     FINISHED MERGING THREE FILES!')
                
                three_classes_counter += 1

        three_class_end = time.time()
        print('FINISHED 3 CLASS MERGE. TIME SPENT:', round(three_class_end - three_class_start, 2))

def saveOneHot():

    subdirs, dirs, files = os.walk(DATA_FILEPATHS).__next__()

    i = 0

    y_train_one_hot = np.zeros((len(files), 397))

    for file in files:

        entries = file.split('.')[0]
        entries = entries.split('_')
        entries = entries[:-1]
        
        y_entry = np.zeros(397)

        for entry in entries:
            
            idx = CLASS_TO_INT[entry]
            
            y_entry[idx] = 1
            
        y_train_one_hot[i] = y_entry
        
        i += 1

    # saving the y_labels_one_hot array as a .npy file
    np.save('y_labels_one_hot.npy', y_train_one_hot)

    # saving the filename array as .npy file
    np.save('filenames.npy', files)

    filenames_shuffled, y_labels_one_hot_shuffled = shuffle(files, y_train_one_hot)
    np.save('y_labels_one_hot_shuffled.npy', y_labels_one_hot_shuffled)
    np.save('filenames_shuffled.npy', filenames_shuffled)

    filenames_shuffled_numpy = np.array(filenames_shuffled)

    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
        filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=0.2, random_state=RANDOM_STATE)   

    np.save('X_train_filenames.npy', X_train_filenames)
    np.save('y_train.npy', y_train)

    np.save('X_val_filenames.npy', X_val_filenames)
    np.save('y_val.npy', y_val)


def trimSoundscapes():

    for subdir, dirs, files in os.walk(DATA_FILEPATHS):

        for file in files:

            file_split = file.split('_')

            sound_label = METADATA[METADATA['audio_id'] == int(file_split[0])]

            path = 'datasets/train/0garbage/train_soundscapes/' + file
            
            audio, _ = librosa.load(path, sr=32000)

            counter = 0
            for i in range(0, len(audio), int(5 * 32000)):
                
                split = audio[i:i + int(5 * 32000)]

                # End of signal?S
                if len(split) < int(5 * 32000):
                    break

                # Save split
                split_dir = 'datasets/train/splitted_train_soundscape/'
                if not os.path.exists(split_dir):
                    os.makedirs(split_dir)

                save_path = (split_dir + str(sound_label['seconds'].iloc[counter]) + '_' + str(sound_label['audio_id'].iloc[counter]) + '_' + sound_label['birds'].iloc[counter])
                np.save(save_path, split)

                counter += 1


def createMelspec():

    for subdir, dirs, files in os.walk(DATA_FILEPATHS):

        for file in files:

            file_path = DATA_FILEPATHS + file
            audio = np.load(file_path)

            hop_length = int(SIGNAL_LENGTH * SAMPLING_RATE / (INPUT_SHAPE[1] - 1))
            mel_spec = librosa.feature.melspectrogram(y=audio,
                                                    sr=SAMPLING_RATE,
                                                    n_fft=N_FFT,
                                                    hop_length=hop_length,
                                                    n_mels=INPUT_SHAPE[0],
                                                    fmin=FMIN,
                                                    fmax=FMAX)
            
            # Save melspectogram as image file
            melspec_save_dir = 'datasets/train/mel_nocalls/'
            if not os.path.exists(melspec_save_dir):
                os.makedirs(melspec_save_dir)
                
            save_path = (melspec_save_dir + file.replace('.npy', ''))
            np.save(save_path, mel_spec)




def removeNocall():

    # Trim train_sounscapes (remove nocall and where 1 > labels)
    train_soundscapes = pd.read_csv('datasets/train/0garbage/train_soundscape_labels.csv')
    birds = train_soundscapes['birds'].value_counts()

    keys = list(birds.keys())
    remove_keys = []
    for key in keys:
        if ' ' in key or key == 'nocall':
            remove_keys.append(key)

    for string in remove_keys:
        train_soundscapes  = train_soundscapes[train_soundscapes['birds'].map(lambda x: str(x)!=string)]
    print(train_soundscapes)

    # Remove soundscapes melspecs where nocall and 1 > labels
    melspec_soundscape_dir = 'datasets/train/collapsed_melspecs_soundscape/'
    removed_dir = 'datasets/train/0garbage/removed_melspecs_soundscape/'

    for subdir, dirs, files in os.walk(melspec_soundscape_dir):

        for file in files:
            
            if 'nocall' in file or ' ' in file:
                shutil.move(melspec_soundscape_dir + file, removed_dir + file)

    for subdir, dirs, files in os.walk(PREPROCESSED_SAVE_DIR):
        col_train = files
    notcol_train = []
    for subdir, dirs, files in os.walk(DATA_FILEPATHS):
        notcol_train.append(files)

    nocol_train = list(itertools.chain.from_iterable(notcol_train))

    files_for_deletion = []
    for file in notcol_train:
        if file not in col_train:
            files_for_deletion.append(file)
    print(files_for_deletion)

def saveNocall():
    
    file_counter = 0

    for subdir, dirs, files in os.walk(DATA_FILEPATHS):
        
        for file in files:

            file_split = file.split('_')

            sound_label = METADATA[METADATA['audio_id'] == int(file_split[0])]
            path = 'datasets/train/0garbage/train_soundscapes/' + file
            
            audio, _ = librosa.load(path, sr=32000)

            counter = 0
            for i in range(0, len(audio), int(5 * 32000)):
                
                split = audio[i:i + int(5 * 32000)]
                
                if sound_label['birds'].iloc[counter] == 'nocall':
                
                    # Save split
                    split_dir = 'datasets/train/nocalls/'
                    if not os.path.exists(split_dir):
                        os.makedirs(split_dir)
                        
                    save_path = (split_dir + sound_label['birds'].iloc[counter] + '_' + str(file_counter))
                    np.save(save_path, split)

                    file_counter += 1
                    
                counter += 1


def trimAudios():

    start_trim = time.time()

    for label in LABELS:

        repo = 'datasets/train/0garbage/train_short_audio/' + str(label)
        audios = os.listdir(repo)

        print('  PROCESSING', label, '...')
        start_label = time.time()

        for audio in audios:

            if audio in FILES:

                start_audio = time.time()

                file_path = repo + '/' + audio

                print("    PROCESSING", file_path, '...')

                sig, rate = librosa.load(file_path, sr=32000, offset=None)

                mean = np.mean(np.abs(sig))
                threshold = (np.max(sig) + 4.5*mean)/2.5

                idxs = np.ravel(np.argwhere(sig >= threshold))

                if len(idxs) == 0:
                    continue

                total_idxs=[[x for x in range(idxs[0] - INDENT_SECONDS , idxs[0] + INDENT_SECONDS + 1, 1)]]
                i = 0

                for idx in idxs[1:]:
                    if idx - INDENT_SECONDS <= total_idxs[i][-1]:
                        local_idxs = [x for x in range(total_idxs[i][-1] + 1, idx + INDENT_SECONDS + 1, 1)]
                    else:
                        local_idxs = [x for x in range(idx - INDENT_SECONDS, idx + INDENT_SECONDS + 1, 1)]
                    total_idxs.append(local_idxs)
                    i += 1

                flat_total_idxs = np.array([item for sublist in total_idxs for item in sublist])
                flat_total_idxs = flat_total_idxs[flat_total_idxs >= 0]
                flat_total_idxs = flat_total_idxs[flat_total_idxs <= len(sig) - 1]
                
                sig_trimmed = np.array([sig[index] for index in flat_total_idxs])
                
                # Save as image file
                save_dir = 'datasets/train/trimmed_train_short_3.5/' + str(label)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = (save_dir + '/' + str(audio))
                save_path = save_path.replace('.ogg', '')
                np.save(save_path, sig_trimmed)

                end_audio = time.time()
                
                print('    FINISHED PROCESSING', file_path, '!')
                print('    TIME ELAPSED FOR', file_path, round(end_audio - start_audio, 2))
                print('    TRIMMED AUDIO SHAPE', sig_trimmed.shape, '\n')

        end_label = time.time()
                
        print('  FINISHED TRIMMING', str(label), '!')
        print('  TIME ELAPSED FOR', label, round(end_label - start_label, 2))
        print('\n')


    end_trim = time.time()
    print('TIME ELAPSED FOR TRIMMING', round(end_trim - start_trim, 2))
