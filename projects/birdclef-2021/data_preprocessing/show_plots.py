import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
from sklearn.utils import shuffle
from PIL import Image

TRAIN_META = pd.read_csv('../datasets/train/train_metadata.csv')
RANDOM_SEED = 1337

labels = TRAIN_META['primary_label'].unique()
ratings = [2, 2.5, 3, 3.5, 4, 4.5, 5]


# PLOT spectograms
plt.figure(figsize=(25, 5))
for i in range(3):
    spec = Image.open('../datasets/train/splitted_trimmed_melsepcs/acafly' + str(i) + '.png')
    plt.subplot(2, 3, i + 1)
    plt.title('acafly/XC31224_' + str(i))
    plt.imshow(spec, origin='lower')

# # PLOT audio graphs
# for label in labels[80:100]:
#     repo = 'train_short_audio/' + str(label)
#     audio = os.listdir(repo)[0]
#     file_path = repo + '/' + audio

#     sig, rate = librosa.load(file_path, sr=32000, offset=None)

#     mean = np.mean(np.abs(sig))
#     threshold = (np.max(sig) - mean)/2.5

#     # Plot the signal
#     plt.figure(figsize=(15, 5))
#     plt.title(file_path)
#     librosa.display.waveplot(sig, sr=32000)
#     plt.hlines(threshold, 0, len(sig),colors='black')
#     plt.hlines(mean, 0, len(sig),colors='black')

# for rating in ratings:
#     TRAIN_META = shuffle(TRAIN_META, random_state=1338)
#     rating_series = TRAIN_META['rating'] == rating
#     rating_idx = rating_series[rating_series].index[0]

#     rating_label = TRAIN_META['primary_label'][rating_idx]
#     rating_file = TRAIN_META['filename'][rating_idx]

#     audio_path = 'train_short_audio/' + str(rating_label) + '/' + str(rating_file)

#     sig, rate = librosa.load(audio_path, sr=32000, offset=None)

#     mean = np.mean(np.abs(sig))
#     threshold = (np.max(sig) + 4.5*mean)/2.5


#     # Plot the signal
#     plt.figure(figsize=(15, 5))
#     plt.title(audio_path + str(rating))
#     librosa.display.waveplot(sig, sr=32000)
#     plt.hlines(threshold, 0, len(sig),colors='black')
#     plt.hlines(mean, 0, len(sig),colors='black')
