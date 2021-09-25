import os
import random
import shutil

train_dir = 'datasets/train/0garbage/mel_train/'
val_dir = 'datasets/train/mel_val/'
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

_, dirs, _ = os.walk(train_dir).__next__()

for dir in dirs:
    files = os.listdir(train_dir + dir + '/')
    n_elements = int(0.2 * len(files))
    random_files = random.sample(files, n_elements)
    
    for random_file in random_files:
        print('Moving', random_file)
        train_path = train_dir + dir + '/' + random_file
        val_path = val_dir + dir + '/'
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        shutil.move(train_path, val_path + random_file)