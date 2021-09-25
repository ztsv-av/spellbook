import pandas as pd
import numpy as np
import os
import random
import librosa
import time
from pydub import AudioSegment
from scipy.io import wavfile

data_dir = 'datasets/train/numpy_3_full/'
root, labels, files = os.walk(data_dir).__next__()


dest_dir = 'datasets/train/wav_numpy_full/'

forest_path = 'datasets/train/0garbage/noise/forest.wav'
forest = AudioSegment.from_file(forest_path, format="wav")
rain_path = 'datasets/train/0garbage/noise/rain.wav'
rain = AudioSegment.from_file(rain_path, format="wav")



def forest_overlay(audio_path, audio_name, dest_repo, numpy_dir):

    audio = AudioSegment.from_file(audio_path, format="wav")

    random_idx = random.randint(0, int(len(forest) * 0.9))
    noise = forest[random_idx:random_idx + len(audio)]

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
    

def rain_overlay(audio_path, audio_name, dest_repo, numpy_dir):

    audio = AudioSegment.from_file(audio_path, format="wav")

    random_idx = random.randint(0, int(len(rain) * 0.9))
    noise = rain[random_idx:random_idx + len(audio)]

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

def rain_forest_overlay(audio_path, audio_name, dest_repo, numpy_dir):

    audio = AudioSegment.from_file(audio_path, format="wav")

    # rain idx
    rain_random_idx = random.randint(0, int(len(rain) * 0.9))
    rain_noise = rain[rain_random_idx:rain_random_idx + len(audio)]

    # forest idx
    forest_random_idx = random.randint(0, int(len(forest) * 0.9))
    forest_noise = forest[forest_random_idx:forest_random_idx + len(audio)]

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


for label in labels:

    print('STARTING', str(label), '!')

    data_repo = data_dir + str(label) + '/'
    dest_repo = dest_dir + str(label) + '/'

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
                
            forest_rain_overlay(wav_path, wav_name, dest_repo, numpy_repo)
            
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

        forest_overlay(wav_path, wav_name, dest_repo, numpy_repo)
        
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

        rain_overlay(wav_path, wav_name, dest_repo, numpy_repo)
        
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

        rain_forest_overlay(wav_path, wav_name, dest_repo, numpy_repo)
        
        os.remove(numpy_path)   # remove original numpy file

    forest_rain_time_end = time.time()
    print(' TIME SPEND FOR FOREST + RAIN AUGMENTATION', round(forest_rain_time_end - forest_rain_time_start, 2))

    print('  FINISHED WITH', str(label), '!')