import numpy as np
import pandas as pd
import os
import librosa

soundscape_dir = 'datasets/train/0garbage/train_soundscapes/'

soundscape_labels = pd.read_csv('datasets/train/0garbage/train_soundscape_labels.csv')
soundscape_labels = soundscape_labels.drop(labels = 'site', axis=1)

file_counter = 0
for subdir, dirs, files in os.walk(soundscape_dir):
    
    for file in files:

        file_split = file.split('_')

        sound_label = soundscape_labels[soundscape_labels['audio_id'] == int(file_split[0])]
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