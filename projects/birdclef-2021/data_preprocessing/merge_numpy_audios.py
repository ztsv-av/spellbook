import pandas as pd
import numpy as np
import os
import librosa
import time
from pydub import AudioSegment
from scipy.io import wavfile

TRAIN_META = pd.read_csv('datasets/train/0garbage/train_metadata.csv')
TRAIN_META = TRAIN_META[TRAIN_META['rating'] >= 3.5]
FILES = list(TRAIN_META['filename'])
LABELS = TRAIN_META['primary_label'].unique()

class_to_int = {k: v for v, k in enumerate(LABELS)}
idxs = list(class_to_int.values())

data_dir = 'datasets/train/trimmed_train_short_3.5/'
wav_dir_2_classes = 'datasets/train/wav_trimmed_train_short_3.5_2classes/'
wav_merged_dir_2_classes = 'datasets/train/wav_merged_trimmed_train_short_3.5_2classes/'
dest_dir_2_classes = 'datasets/train/trimmed_merged_train_short_3.5_2classes/'
wav_dir_3_classes = 'datasets/train/wav_trimmed_train_short_3.5_3classes/'
wav_merged_dir_3_classes = 'datasets/train/wav_merged_trimmed_train_short_3.5_3classes/'
dest_dir_3_classes = 'datasets/train/trimmed_merged_train_short_3.5_3classes/'

# two_classes_len = len(FILES) * 0.3
three_classes_len = len(FILES)

# print('STARTING 2 CLASS MERGE...')
# two_class_start = time.time()
# two_classes_counter = 0
# while two_classes_counter < two_classes_len:
    
#     first_class_int = int(np.random.uniform(low=0, high=396))
#     second_class_int = int(np.random.uniform(low=0, high=396))
#     if first_class_int == second_class_int:
#         continue
    
#     first_class = list(class_to_int.keys())[list(class_to_int.values()).index(first_class_int)]
#     second_class = list(class_to_int.keys())[list(class_to_int.values()).index(second_class_int)]
    
#     first_class_files = os.listdir(data_dir + first_class + '/')
#     second_class_files = os.listdir(data_dir + second_class + '/')
    
#     first_class_random_filename = np.random.choice(first_class_files)
#     second_class_random_filename = np.random.choice(second_class_files)
    
#     if (os.path.exists(dest_dir_2_classes + first_class + '_' + second_class + '/' + first_class + '_' + second_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename) or 
#         os.path.exists(dest_dir_2_classes + second_class + '_' + first_class + '/' + second_class + '_' + first_class + '_' + second_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename)):
#         continue
    
#     print('     MERGING', first_class + '/' + first_class_random_filename, second_class + '/' + second_class_random_filename)
    
#     first_class_numpy_file = np.load(data_dir + first_class + '/' + first_class_random_filename)
#     first_class_numpy = np.int16(first_class_numpy_file/np.max(np.abs(first_class_numpy_file)) * 32767)
#     second_class_numpy_file = np.load(data_dir + second_class + '/' + second_class_random_filename)
#     second_class_numpy = np.int16(second_class_numpy_file/np.max(np.abs(second_class_numpy_file)) * 32767)
    
#     first_class_wav_path = wav_dir_2_classes + first_class + '/' + first_class_random_filename.replace('npy', 'wav')
#     if not os.path.exists(wav_dir_2_classes + first_class + '/'):
#         os.makedirs(wav_dir_2_classes + first_class + '/')
#     second_class_wav_path = wav_dir_2_classes + second_class + '/' + second_class_random_filename.replace('npy', 'wav')
#     if not os.path.exists(wav_dir_2_classes + second_class + '/'):
#         os.makedirs(wav_dir_2_classes + second_class + '/')
    
#     wavfile.write(first_class_wav_path, 32000, first_class_numpy)
#     wavfile.write(second_class_wav_path, 32000, second_class_numpy)
    
#     first_class_wav = AudioSegment.from_file(first_class_wav_path, format="wav")
#     second_class_wav = AudioSegment.from_file(second_class_wav_path, format="wav")
    
#     if len(first_class_numpy) <= len(second_class_numpy):
#         overlay = first_class_wav.overlay(second_class_wav, position=0)
#     else:
#         overlay = second_class_wav.overlay(first_class_wav, position=0)
    
#     wav_merged_path_2_classes = (wav_merged_dir_2_classes + first_class + '_' + second_class + '/' + 
#                                 first_class + '_' + second_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('npy', 'wav'))
    
#     if not os.path.exists(wav_merged_dir_2_classes + first_class + '_' + second_class + '/'):
#         os.makedirs(wav_merged_dir_2_classes + first_class + '_' + second_class + '/')
    
#     overlay.export(wav_merged_path_2_classes, format="wav")
    
#     sig_int, rate = librosa.load(wav_merged_path_2_classes, sr=32000, offset=None)
#     sig_16 = (sig_int* 32767).astype(int)
#     sig_float = sig_16/32767.0

#     numpy_merged_path_2_classes = (dest_dir_2_classes + first_class + '_' + second_class + '/' + 
#                                   first_class + '_' + second_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename)
#     if not os.path.exists(dest_dir_2_classes + first_class + '_' + second_class + '/'):
#         os.makedirs(dest_dir_2_classes + first_class + '_' + second_class + '/')
#     np.save(numpy_merged_path_2_classes, sig_float)
    
#     print('     FINISHED MERGING TWO FILES!')
    
#     two_classes_counter += 1
# two_class_end = time.time()
# print('FINISHED 2 CLASS MERGE. TIME SPENT:', round(two_class_end - two_class_start, 2))

print('STARTING 3 CLASS MERGE...')
three_class_start = time.time()
three_classes_counter = 0
while three_classes_counter < three_classes_len:
    
    for first_class_int in idxs:
    
        second_class_int = int(np.random.uniform(low=0, high=396))
        third_class_int = int(np.random.uniform(low=0, high=396))
        if first_class_int == second_class_int or first_class_int == third_class_int or second_class_int == third_class_int:
            continue
        
        first_class = list(class_to_int.keys())[list(class_to_int.values()).index(first_class_int)]
        second_class = list(class_to_int.keys())[list(class_to_int.values()).index(second_class_int)]
        third_class = list(class_to_int.keys())[list(class_to_int.values()).index(third_class_int)]
        
        first_class_files = os.listdir(data_dir + first_class + '/')
        second_class_files = os.listdir(data_dir + second_class + '/')
        third_class_files = os.listdir(data_dir + third_class + '/')
        
        first_class_random_filename = np.random.choice(first_class_files)
        second_class_random_filename = np.random.choice(second_class_files)
        third_class_random_filename = np.random.choice(third_class_files)
        
        if (os.path.exists(dest_dir_3_classes + first_class + '_' + second_class + '_' + third_class + '/' + first_class + '_' + second_class + '_' + third_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename) or
            os.path.exists(dest_dir_3_classes + first_class + '_' + third_class + '_' + second_class + '/' + first_class + '_' + third_class + '_' + second_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename) or 
            os.path.exists(dest_dir_3_classes + second_class + '_' + first_class + '_' + third_class + '/' + second_class + '_' + first_class + '_' + third_class + '_' + second_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename) or
            os.path.exists(dest_dir_3_classes + second_class + '_' + third_class + '_' + first_class + '/' + second_class + '_' + third_class + '_' + first_class + '_' + second_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename) or
            os.path.exists(dest_dir_3_classes + third_class + '_' + first_class + '_' + second_class + '/' + third_class + '_' + first_class + '_' + second_class + '_' + third_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename) or
            os.path.exists(dest_dir_3_classes + third_class + '_' + second_class + '_' + first_class + '/' + third_class + '_' + second_class + '_' + first_class + '_' + third_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('.npy', '') + '_' + first_class_random_filename)):
            continue
        
        print('     MERGING', first_class + '/' + first_class_random_filename, second_class + '/' + second_class_random_filename, third_class + '/' + third_class_random_filename)
        
        first_class_numpy_file = np.load(data_dir + first_class + '/' + first_class_random_filename)
        first_class_numpy = np.int16(first_class_numpy_file/np.max(np.abs(first_class_numpy_file)) * 32767)
        second_class_numpy_file = np.load(data_dir + second_class + '/' + second_class_random_filename)
        second_class_numpy = np.int16(second_class_numpy_file/np.max(np.abs(second_class_numpy_file)) * 32767)
        third_class_numpy_file = np.load(data_dir + third_class + '/' + third_class_random_filename)
        third_class_numpy = np.int16(third_class_numpy_file/np.max(np.abs(third_class_numpy_file)) * 32767)
        
        first_class_wav_path = wav_dir_3_classes + first_class + '/' + first_class_random_filename.replace('npy', 'wav')
        if not os.path.exists(wav_dir_3_classes + first_class + '/'):
            os.makedirs(wav_dir_3_classes + first_class + '/')
        second_class_wav_path = wav_dir_3_classes + second_class + '/' + second_class_random_filename.replace('npy', 'wav')
        if not os.path.exists(wav_dir_3_classes + second_class + '/'):
            os.makedirs(wav_dir_3_classes + second_class + '/')
        third_class_wav_path = wav_dir_3_classes + third_class + '/' + third_class_random_filename.replace('npy', 'wav')
        if not os.path.exists(wav_dir_3_classes + third_class + '/'):
            os.makedirs(wav_dir_3_classes + third_class + '/')
        
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
        
        wav_merged_path_3_classes = wav_merged_dir_3_classes + first_class + '_' + second_class + '_' + third_class + '/' + first_class + '_' + second_class + '_' + third_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename.replace('npy', 'wav')
        
        if not os.path.exists(wav_merged_dir_3_classes + first_class + '_' + second_class + '_' + third_class + '/'):
            os.makedirs(wav_merged_dir_3_classes + first_class + '_' + second_class + '_' + third_class + '/')
        
        overlay2.export(wav_merged_path_3_classes, format="wav")
        
        sig_int, rate = librosa.load(wav_merged_path_3_classes, sr=32000, offset=None)
        sig_16 = (sig_int* 32767).astype(int)
        sig_float = sig_16/32767.0

        numpy_merged_path_3_classes = dest_dir_3_classes + first_class + '_' + second_class + '_' + third_class + '/' + first_class + '_' + second_class + '_' + third_class + '_' + first_class_random_filename.replace('.npy', '') + '_' + second_class_random_filename.replace('.npy', '') + '_' + third_class_random_filename
        if not os.path.exists(dest_dir_3_classes + first_class + '_' + second_class + '_' + third_class + '/'):
            os.makedirs(dest_dir_3_classes + first_class + '_' + second_class + '_' + third_class + '/')
        np.save(numpy_merged_path_3_classes, sig_float)
        
        print('     FINISHED MERGING THREE FILES!')
        
        three_classes_counter += 1
three_class_end = time.time()
print('FINISHED 3 CLASS MERGE. TIME SPENT:', round(three_class_end - three_class_start, 2))