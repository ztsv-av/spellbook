import pandas as pd
import numpy as np
import librosa
import librosa.display
import warnings
import os
import time
from sklearn.utils import shuffle
from PIL import Image

from GLOBAL_VARS import SAMPLE_RATE, SIGNAL_LENGTH, SPEC_SHAPE, FMIN, FMAX

TRAIN_META = pd.read_csv('datasets/train/0garbage/train_metadata.csv')

train_dir = 'datasets/train/trimmed_numpy_train/'
root, labels, files = os.walk(train_dir).__next__()


print('PROCESSING MERGING, SPLITTING AND CREATING MELSPECTROGRAMS...\n')
start_operation = time.time()

for label in labels:

    # Concantenate audios of one class into one big audio
    
    full_audio = np.array([])

    repo = train_dir + label
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

        # # Save split
        # split_save_dir = 'datasets/train/splitted_numpy_train/' + label
        # if not os.path.exists(split_save_dir):
        #     os.makedirs(split_save_dir)
        # save_path = (split_save_dir + '/' + label + '_' + str(file_count))
        # np.save(save_path, split)

        # Create melspectogram of 5 second split

        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=split, 
                                                  sr=SAMPLE_RATE, 
                                                  n_fft=1024, 
                                                  hop_length=hop_length, 
                                                  n_mels=SPEC_SHAPE[0], 
                                                  fmin=FMIN, 
                                                  fmax=FMAX)
    
        #mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        #mel_spec = (mel_spec + 80) / 80 
        
        # mel_spec -= mel_spec.min()
        # mel_spec /= mel_spec.max()
        # if not mel_spec.max() == 1.0 or not mel_spec.min() == 0.0:
        #     continue

        # Save melspectogram
        melspec_save_dir = 'datasets/train/mel_train/' + label
        if not os.path.exists(melspec_save_dir):
            os.makedirs(melspec_save_dir)
        save_path = (melspec_save_dir + '/' + label + '_' + str(file_count))
        #im = Image.fromarray(mel_spec * 255.0).convert("L")
        #im.save(save_path)
        np.save(save_path, mel_spec)

        file_count += 1

    end_melspec = time.time()
    print('   FINISHED WITH MELSPECTOGRAMS OF', label, '!')
    print('   TIME ELAPSED FOR MELSPECTOGRAMS OF', label, round(end_melspec - start_melspec, 2), '\n')

end_operation = time.time()
print('   FINISHED WITH OPERATION!')
print('   TIME ELAPSED:', round(end_operation - start_operation, 2))

