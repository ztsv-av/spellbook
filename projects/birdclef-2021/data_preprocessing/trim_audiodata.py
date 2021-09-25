import numpy as np
import pandas as pd
import os
import librosa
import time

from GLOBAL_VARS import INDENT_SEC

TRAIN_META = pd.read_csv('datasets/train/0garbage/train_metadata.csv')
TRAIN_META = TRAIN_META[TRAIN_META['rating'] >= 3.5]

labels = TRAIN_META['primary_label'].unique()

files = list(TRAIN_META['filename'])

# TRIM AUDIO
start_trim = time.time()

for label in labels:
    repo = 'datasets/train/0garbage/train_short_audio/' + str(label)
    audios = os.listdir(repo)

    print('  PROCESSING', label, '...')
    start_label = time.time()

    for audio in audios:

        if audio in files:

            start_audio = time.time()

            file_path = repo + '/' + audio

            print("    PROCESSING", file_path, '...')

            sig, rate = librosa.load(file_path, sr=32000, offset=None)

            mean = np.mean(np.abs(sig))
            threshold = (np.max(sig) + 4.5*mean)/2.5

            idxs = np.ravel(np.argwhere(sig >= threshold))

            if len(idxs) == 0:
                continue

            total_idxs=[[x for x in range(idxs[0] - INDENT_SEC , idxs[0] + INDENT_SEC + 1, 1)]]
            i = 0

            for idx in idxs[1:]:
                if idx - INDENT_SEC <= total_idxs[i][-1]:
                    local_idxs = [x for x in range(total_idxs[i][-1] + 1, idx + INDENT_SEC + 1, 1)]
                else:
                    local_idxs = [x for x in range(idx - INDENT_SEC, idx + INDENT_SEC + 1, 1)]
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