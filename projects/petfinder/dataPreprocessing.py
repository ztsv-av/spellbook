import tensorflow as tf

import os
import pandas as pd
import numpy as np
import time
import shutil
from tqdm.notebook import tqdm

os.chdir('V:\Git\spellbook')

from permutationFunctions import classification_permutations
from preprocessFunctions import minMaxNormalizeNumpy, addColorChannels
from helpers import loadImage, saveNumpyArray, visualizeImageBox, loadNumpy, splitTrainValidation, evaluateString


def preprocessData():

    ids = NEW_METADATA['id'].values

    id_list = []
    no_id_list = []
    wrong_shape_images_list = []

    for path in DATA_PATHS:

        image_id = path.split('/')[-1].replace('.jpg', '').split('-')[0]

        if image_id in id_list:
            continue
        else:
            id_list.append(image_id)

        if image_id not in ids:
            continue

        score = METADATA[METADATA['PetID'] == image_id]['AdoptionSpeed'].values
        if len(score) == 0:
            no_id_list.append(image_id)
            continue
        else:
            score = score[0]

        image = loadImage(path, image_type='uint8')

        if len(image.shape) != 3:
            wrong_shape_images_list.append(image_id)
            continue

        image = tf.image.resize(image, RESHAPE_SIZE)
        image = tf.cast(image, tf.uint8).numpy()

        filename = image_id + '_' + str(score)
        saveNumpyArray(image, SAVE_DATA_DIR + filename)


def preprocessMetadata():

    breeds_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-new-breeds.csv')

    new_metadata = pd.DataFrame(
        columns=[
            'Type', 'Age', 'Breed', 'Gender', 'Color', 'Maturity', 'Fur',
            'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'id', 'AdoptionSpeed'])
    for index, row in METADATA.iterrows():

        animal_type = row[0]
        one_hot_type = [0 for val in range(2)]
        one_hot_type[animal_type - 1] = 1

        animal_age = row[2]



        animal_breed1 = row[3]
        if (animal_breed1 == 0) or (animal_breed1 == 307):
            continue

        one_hot_breed1 = [0 for val in range(307)]
        one_hot_breed1[animal_breed1 - 1] = 1

        # animal_breed2 = row[4]
        # one_hot_breed2 = np.zeros((308))
        # one_hot_breed2[animal_breed2 - 1] = 1

        animal_gender = row[5]
        if animal_gender == 3:
            continue
        one_hot_gender = [0 for val in range(2)]
        one_hot_gender[animal_gender - 1] = 1

        animal_color1 = row[6]
        animal_color2 = row[7]
        animal_color3 = row[8]
        animal_color = [animal_color1, animal_color2, animal_color3]
        one_hot_color = [0 for val in range(7)]
        for color in animal_color:
            one_hot_color[color - 1] = 1

        animal_maturity = row[9]
        one_hot_maturity = [0 for val in range(5)]
        one_hot_maturity[animal_maturity - 1] = 1

        animal_fur = row[10]
        one_hot_fur = [0 for val in range(4)]
        one_hot_fur[animal_fur - 1] = 1

        animal_vaccinated = row[11]
        one_hot_vaccinated = [0 for val in range(3)]
        one_hot_vaccinated[animal_vaccinated - 1] = 1

        animal_dewormed = row[12]
        one_hot_dewormed = [0 for val in range(3)]
        one_hot_dewormed[animal_dewormed - 1] = 1

        animal_sterilized = row[13]
        one_hot_sterilized = [0 for val in range(3)]
        one_hot_sterilized[animal_sterilized - 1] = 1

        animal_health = row[14]
        one_hot_health = [0 for val in range(3)]
        one_hot_health[animal_health - 1] = 1

        animal_quantity = row[15]
        if animal_quantity > 1:
            continue

        # animal_fee = row[16]
        # one_hot_fur = np.zeros((4))
        # one_hot_fur[animal_fur - 1] = 1

        # animal_state = row[17]
        # one_hot_fur = np.zeros((4))
        # one_hot_fur[animal_fur - 1] = 1

        # animal_video = row[19]
        # one_hot_fur = np.zeros((4))
        # one_hot_fur[animal_fur - 1] = 1

        # animal_description = row[20]

        animal_id = row[21]

        animal_adoption_speed = row[23]

        # animal_photo = row[22]

        new_metadata = new_metadata.append({'Type': one_hot_type, 'Age': animal_age, 'Breed': one_hot_breed1,
            'Gender': one_hot_gender, 'Color': one_hot_color, 'Maturity': one_hot_maturity,
            'Fur': one_hot_fur, 'Vaccinated': one_hot_vaccinated, 'Dewormed': one_hot_dewormed,
            'Sterilized': one_hot_sterilized, 'Health': one_hot_health, 'id': animal_id,
            'AdoptionSpeed': animal_adoption_speed}, ignore_index=True)

    new_metadata.to_csv(path_or_buf=NEW_METADATA_SAVE_PATH, index=False)


def prepareCats():

    cats_path = 'projects/petfinder/petfinder-previous/data/additional-data/cats-images/'
    cats_dirs = os.listdir(cats_path)

    for cat_dir in cats_dirs:

        # just use .replace you dumbass

        cat_path = cats_path + cat_dir
        cat_dir = cat_dir.split(' ')

        proper_name = ''

        for name in cat_dir:

            proper_name += name.lower()

        proper_path = cats_path + proper_name
        os.rename(cat_path, proper_path)


def prepareDogs():

    dogs_path = 'projects/petfinder/petfinder-previous/data/additional-data/dogs-images/'
    dogs_dirs = os.listdir(dogs_path)

    for dog_dir in os.listdir(dogs_dirs):

        # just use .replace you dumbass

        dog_path = dogs_path + dog_dir
        dog_name = dog_dir.split('-')[1:]

        for idx, name in enumerate(dog_name):

            splitted_name = name.split('_')

            dog_name[idx] = ''
            for string in splitted_name:
                dog_name[idx] += string.lower()

        proper_name = ''

        for string in dog_name:

            proper_name += string
            
        proper_path = dogs_path + proper_name
        os.rename(dog_path, proper_path)


def createCatsDogsMeta():

    cats_path = 'projects/petfinder/petfinder-previous/data/additional-data/cats-images/'
    dogs_path = 'projects/petfinder/petfinder-previous/data/additional-data/dogs-images/'

    cats_dogs_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/cats-dogs-metadata.csv')

    breeds_meta_list = []
    breeds_meta_save_path = 'projects/petfinder/petfinder-previous/data/additional-data/breeds-metadata.csv'

    for breed_dir in os.listdir(cats_path):

        for image_name in os.listdir(cats_path + breed_dir):

            breeds_meta_list.append([image_name, [1, 0]])

    for breed_dir in os.listdir(dogs_path):
    
        for image_name in os.listdir(dogs_path + breed_dir):

            breeds_meta_list.append([image_name, [0, 1]])

    breeds_meta = pd.DataFrame(breeds_meta_list, columns=['id', 'type'])

    full_breeds_meta = breeds_meta.append(cats_dogs_meta, ignore_index=True)

    full_breeds_meta.to_csv(path_or_buf=breeds_meta_save_path, index=False)



def preprocessPetfinderBreeds():

    meta_path = 'projects/petfinder/petfinder-previous/data/metadata/petfinder-breed-labels.csv'
    petfinder_breeds_meta = pd.read_csv(meta_path)

    new_meta_path = 'projects/petfinder/petfinder-previous/data/metadata/petfinder-new-breeds.csv'
    mixed_breed_meta_path = 'projects/petfinder/petfinder-previous/data/metadata/petfinder-mixed-breeds.csv'

    new_meta_list = []
    mixed_breed_meta_list = []

    for index, row in petfinder_breeds_meta.iterrows():
        
        animal_breed_id = row['BreedID']
        animal_type = [1, 0] if row['Type'] == 2 else [0, 1]
        animal_breed = row['BreedName']

        animal_breed = animal_breed.lower()
        animal_breed = animal_breed.replace(' ', '')
        animal_breed = animal_breed.replace('-', '')

        # mixed breed
        if animal_breed_id == 307:

            mixed_breed_meta_list.append([animal_breed_id, animal_type, animal_breed])

        else:

            new_meta_list.append([animal_breed_id, animal_type, animal_breed])

        mixed_breed_meta = pd.DataFrame(mixed_breed_meta_list, columns=['id', 'type', 'breed'])
        new_meta = pd.DataFrame(new_meta_list, columns=['id', 'type', 'breed'])
    
        mixed_breed_meta.to_csv(path_or_buf=new_meta_path, index=False)
        new_meta.to_csv(path_or_buf=mixed_breed_meta_path, index=False)


def savePetfinderBreeds():

    images_dir = 'projects/petfinder/petfinder-previous/data/images/'
    metadata = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/metadata.csv')

    breeds_metadata = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-ids.csv')

    petfinder_breeds_meta_list = []
    petfinder_breeds_meta_save_path = 'projects/petfinder/petfinder-previous/data/metadata/breeds.csv'

    for image_name in os.listdir(images_dir):

        image_path = images_dir + image_name

        image_id = image_name.split('-')[0]

        breed_id = metadata[metadata['PetID'] == image_id]['Breed1'].values

        if len(breed_id) == 0:
            continue
        else:
            breed_id = breed_id[0]
        
        if (breed_id == 0) or (breed_id == 307):
            continue

        animal_type = breeds_metadata[breeds_metadata['id'] == breed_id]['type'].values[0]
        breed_name = breeds_metadata[breeds_metadata['id'] == breed_id]['breed'].values[0]

        if (breed_name == 'domesticshorthair') or (breed_name == 'domesticmediumhair') or (breed_name == 'domesticlonghair'):
            continue

        if (breed_name == 'tiger'):

            breed_name = 'tabby'

        if (breed_name == 'korat'):

            breed_name = 'russianblue'

        if (breed_name == 'chartreux'):

            breed_name = 'britishshorthair'

        if (breed_name == 'siberian'):

            breed_name = 'norwegianforestcat'

        if (breed_name == 'ragamuffin'):

            breed_name = 'norwegianforestcat'

        if (breed_name == 'tonkinese'):

            breed_name = 'siamese'

        if (breed_name == 'appleheadsiamese'):

            breed_name = 'siamese'

        breed_save_dir = 'projects/petfinder/petfinder-previous/data/petfinder-breeds-images/' + breed_name + '/'
        if not os.path.exists(breed_save_dir):
            os.makedirs(breed_save_dir)
        
        image_save_path = breed_save_dir + image_name

        petfinder_breeds_meta_list.append([image_id, animal_type])
        
        shutil.copy(image_path, image_save_path)
    
    petfinder_breeds_meta = pd.DataFrame(petfinder_breeds_meta_list, columns=['id', 'type'])
    petfinder_breeds_meta.to_csv(path_or_buf=petfinder_breeds_meta_save_path, index=False)


def createBreedIdsMeta():

    data_dir = 'projects/petfinder/petfinder-previous/data/additional-data/breeds-images/'
    breeds = os.listdir(data_dir)

    breeds_ids = [id for id in range(1, len(breeds) + 1)]

    breed_id_meta_list = []

    breed_id_meta_save_path = 'projects/petfinder/petfinder-previous/data/additional-data/breeds-ids.csv'

    for breed_id, breed in zip(breeds_ids, breeds):

        breed_id_meta_list.append([breed_id, breed])
    
    breed_id_meta = pd.DataFrame(breed_id_meta_list, columns=['id', 'breed'])

    breed_id_meta.to_csv(path_or_buf=breed_id_meta_save_path, index=False)


def createBreedsMeta():

    data_dir = 'projects/petfinder/petfinder-previous/data/additional-data/breeds-images/'
    breeds = os.listdir(data_dir)
    breeds_metadata = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/breeds.csv')
    breeds_ids_metadata = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/breeds-ids.csv')

    preprocessed_metadata_list = []
    preprocessed_metadata_save_path = 'projects/petfinder/petfinder-previous/data/additional-data/preprocessed-breeds.csv'

    for breed in breeds:

        breed_path = data_dir + breed

        breed_images = os.listdir(breed_path)

        for image_name in breed_images:
            
            animal_type = breeds_metadata[breeds_metadata['id'] == image_name]['type'].values[0]
            breed_id = breeds_ids_metadata[breeds_ids_metadata['breed'] == breed]['id'].values[0]

            one_hot_breed = [0 for val in range(len(breeds))]
            one_hot_breed[breed_id - 1] = 1

            preprocessed_metadata_list.append([image_name, animal_type, one_hot_breed])
    
    preprocessed_metadata = pd.DataFrame(preprocessed_metadata_list, columns=['id', 'type', 'breed'])
    preprocessed_metadata.to_csv(path_or_buf=preprocessed_metadata_save_path, index=False)


def preprocessBreedImages():

    data_dir = 'projects/petfinder/petfinder-previous/data/additional-data/breeds-images/'
    breeds = os.listdir(data_dir)

    save_data_dir = 'projects/petfinder/petfinder-previous/data/additional-data/breeds-images-preprocessed/'

    grey_images_list = []

    for breed in breeds:

        breed_path = data_dir + breed + '/'

        breed_images = os.listdir(breed_path)

        for image_name in breed_images:

            image_path = breed_path + image_name

            image = loadImage(image_path, image_type='uint8')

            if len(image.shape) < 3:
                grey_images_list.append(image_path)
                continue
            
            image = tf.image.resize(image, RESHAPE_SIZE)
            image = tf.cast(image, tf.uint8).numpy()

            filename = image_name.split('.')[0]
            saveNumpyArray(image, save_data_dir + breed + '_' + filename)


def combineBreedsFurMaturity():

    breeds_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/preprocessed-breeds.csv')
    fur_maturity_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/breeds-fur-size.csv')

    new_meta_list = []

    for index, row in breeds_meta.iterrows():

        pet_id = row['id']
        pet_type = row['type']
        pet_breed = row['breed']

        breed = row['id'].split('_')[0]
        pet_fur = fur_maturity_meta[fur_maturity_meta['breed'] == breed]['FurLength'].values[0]
        pet_fur_onehot = [0, 0, 0]
        pet_fur_onehot[pet_fur - 1] = 1
        pet_size = fur_maturity_meta[fur_maturity_meta['breed'] == breed]['MaturitySize'].values[0]
        pet_size_onehot = [0, 0, 0, 0]
        pet_size_onehot[pet_size - 1] = 1

        new_meta_list.append([pet_id, pet_type, pet_breed, pet_fur_onehot, pet_size_onehot])
    
    new_meta = pd.DataFrame(new_meta_list, columns=['id', 'type', 'breed', 'fur', 'size'])
    new_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/additional-data/breeds-full.csv', index=False)


def preprocessCatsAge():

    cats_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/cats-metadata.csv')
    cats_dir = 'projects/petfinder/petfinder-previous/data/additional-data/cats-images/'

    cats_age_list = []
    cats_save_dir = 'projects/petfinder/petfinder-previous/data/additional-data/cats-age-preprocessed/'

    for breed in os.listdir(cats_dir):

        breed_path = cats_dir + breed + '/'

        for filename in os.listdir(breed_path):

            image_path = breed_path + filename
            
            animal_id = filename.split('_')[0]
            if '.' in animal_id:
                print(image_path)
            else:
                animal_id = evaluateString(animal_id)
            animal_age = cats_meta[cats_meta['id'] == animal_id]['age'].values

            if len(animal_age) == 0:
                continue
            else:
                animal_age = animal_age[0]
            
            animal_age_onehot = [0, 0, 0, 0]
            
            if animal_age == 'Baby':
                animal_age_onehot = [1, 0, 0, 0]
            elif animal_age == 'Young':
                animal_age_onehot = [0, 1, 0, 0]
            elif animal_age == 'Adult':
                animal_age_onehot = [0, 0, 1, 0]
            elif animal_age == 'Senior':
                animal_age_onehot = [0, 0, 0, 1]
            else:
                print(image_path)
                print('No Age ! ! !')
            
            image = loadImage(image_path, image_type='uint8')
            if len(image.shape) != 3:
                continue
            image = tf.image.resize(image, RESHAPE_SIZE)
            image = tf.cast(image, tf.uint8).numpy()
            saveNumpyArray(image, cats_save_dir + breed + '_' + filename.replace('.jpg', ''))

            filename_numpy = filename.replace('.jpg', '.npy')
            cats_age_list.append([breed + '_' + filename_numpy, animal_age_onehot])
    
    cats_age_meta = pd.DataFrame(cats_age_list, columns=['id', 'age'])
    cats_age_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/additional-data/cats-age.csv', index=False)


def typeBreedMeta():

    prep_breeds_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/preprocessed-breeds.csv')

    breed_type_list = []
    animal_breeds = []

    for index, row in prep_breeds_meta.iterrows():

        animal_breed = row['id'].split('_')[0]

        if animal_breed in animal_breeds:

            continue

        else:

            animal_breeds.append(animal_breed)

            animal_type = row['type']
            animal_breed_list = row['breed']
            breed_type_list.append([animal_type, animal_breed_list])

    breed_type_metadata = pd.DataFrame(breed_type_list, columns=['type', 'breed'])
    breed_type_metadata.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/additional-data/type-breed.csv', index=False)


def preprocessPetfinderAge():

    petfinder_images_dir = 'projects/petfinder/petfinder-previous/data/images-preprocessed/256/'
    petfinder_age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/preprocessed_metadata.csv')

    petfinder_age_list = []

    for filename in os.listdir(petfinder_images_dir):

        animal_id = filename.split('_')[0]
        animal_age = petfinder_age_meta[petfinder_age_meta['id'] == animal_id]['Age'].values[0]

        animal_age_onehot = [0, 0, 0, 0]
        if (animal_age >= 0) & (animal_age <= 2):
            animal_age_onehot = [1, 0, 0, 0, 0]
        elif (animal_age >= 3) & (animal_age <= 6):
            animal_age_onehot = [0, 1, 0, 0, 0]
        elif (animal_age >= 7) & (animal_age <= 11):
            animal_age_onehot = [0, 0, 1, 0, 0]
        elif (animal_age >= 12) & (animal_age <= 24):
            animal_age_onehot = [0, 0, 0, 1, 0]
        elif (animal_age >= 25):
            animal_age_onehot = [0, 0, 0, 0, 1]
        else:
            print(filename)
            break

        petfinder_age_list.append([filename, animal_age_onehot])
    
    petfinder_age_meta = pd.DataFrame(petfinder_age_list, columns=['id', 'age'])
    petfinder_age_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/age-5classes-petfinder.csv', index=False)


def combineMetas():

    cats_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/cats-age.csv')
    petfinder_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/age.csv')

    full_meta = pd.concat([cats_meta, petfinder_meta])

    full_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/preprocessed-age.csv', index=False)


def combineAgeWOtherFeatures():

    age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/age-5classes-petfinder.csv')
    pred_breeds_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/predicted-breeds.csv')
    breed_type_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/breeds-type.csv')
    fur_size_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/breeds-fur-size.csv')
    breed_full_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/breeds-full.csv')

    new_age_meta_list = []

    for index, row in age_meta.iterrows():

        animal_id = row['id']
        animal_age = row['age']

        if (len(animal_id.split('_')) > 2):

            animal_type = breed_full_meta[breed_full_meta['id'] == animal_id]['type'].values[0]
            animal_breed = breed_full_meta[breed_full_meta['id'] == animal_id]['breed'].values[0]
            animal_size = breed_full_meta[breed_full_meta['id'] == animal_id]['size'].values[0]
            animal_fur = breed_full_meta[breed_full_meta['id'] == animal_id]['fur'].values[0]
        
        else:

            animal_breed = pred_breeds_meta[pred_breeds_meta['id'] == animal_id]['breed'].values[0]
            animal_breed = animal_breed.replace('\n', '').replace(' ', ', ')

            animal_type = breed_type_meta[breed_type_meta['breed'] == animal_breed]['type'].values[0]
            animal_size = breed_full_meta[breed_full_meta['breed'] == animal_breed]['size'].values[0]
            animal_fur = breed_full_meta[breed_full_meta['breed'] == animal_breed]['fur'].values[0]

        new_age_meta_list.append([animal_id, animal_type, animal_age, animal_breed, animal_size, animal_fur])

    petfinder_age_meta = pd.DataFrame(new_age_meta_list, columns=['id', 'type', 'age', 'breed', 'size', 'fur'])
    petfinder_age_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/age-petfinder-other-features.csv', index=False)


def preprocessDogsAge():

    dir_path = 'projects/petfinder/petfinder-previous/data/additional-data/dogs-age-dataset/'
    save_image_dir = 'projects/petfinder/petfinder-previous/data/additional-data/dogs-age-preprocessed/'

    age_meta_list = []

    for age in os.listdir(dir_path):

        age_path = dir_path + age + '/'

        if age == 'Young':

            age_onehot = [1, 0, 0]

        if age == 'Adult':

            age_onehot = [0, 1, 0]

        else:

            age_onehot = [0, 0, 1]

        for image_name in os.listdir(age_path):

            image_path = age_path + image_name

            image = loadImage(image_path, image_type='uint8')

            if len(image.shape) != 3:
                continue

            image = tf.image.resize(image, RESHAPE_SIZE)
            image = tf.cast(image, tf.uint8).numpy()

            filename = image_name.split('.')[0]
            filename += '.npy'
            saveNumpyArray(image, save_image_dir + filename)

            age_meta_list.append([filename, age_onehot])
    
    age_meta = pd.DataFrame(age_meta_list, columns=['id', 'age'])
    age_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/additional-data/dogs-age.csv', index=False)


def combineAge():

    dogs_dir = 'projects/petfinder/petfinder-previous/data/additional-data/dogs-age-preprocessed/'
    dogs_age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/dogs-age.csv')

    cats_dir = 'projects/petfinder/petfinder-previous/data/additional-data/cats-age-preprocessed/'
    cats_age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/cats-age.csv')
    
    petfinder_dir = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed'
    petfinder_age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-preprocessed-metadata.csv')

    new_age_meta_list = []

    for image_name in tqdm(os.listdir(dogs_dir), desc='Dogs'):

        animal_id = dogs_age_meta[dogs_age_meta['id'] == image_name]['id'].values[0]
        animal_age = dogs_age_meta[dogs_age_meta['id'] == image_name]['age'].values[0]
        animal_age = evaluateString(animal_age)
        animal_age.insert(0, 0)

        new_age_meta_list.append([image_name, animal_age])
    
    for image_name in tqdm(os.listdir(cats_dir), desc='Cats'):

        animal_id = cats_age_meta[cats_age_meta['id'] == image_name]['id'].values[0]
        animal_age = cats_age_meta[cats_age_meta['id'] == image_name]['age'].values[0]

        new_age_meta_list.append([image_name, animal_age])
    
    for image_name in tqdm(os.listdir(petfinder_dir), desc='Petfinder'):

        animal_id = petfinder_age_meta[petfinder_age_meta['id'] == image_name.split('_')[0]]['id'].values[0]
        animal_age = petfinder_age_meta[petfinder_age_meta['id'] == image_name.split('_')[0]]['Age'].values[0]

        if animal_age > 6:
            continue

        animal_age = [1, 0, 0, 0]

        new_age_meta_list.append([image_name, animal_age])

    age_meta = pd.DataFrame(new_age_meta_list, columns=['id', 'age'])
    age_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/additional-data/all-ages.csv', index=False)


def combineAgeBreeds():

    breeds_full_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/breeds-full.csv')
    breeds_petfinder_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/predicted-breeds.csv')
    age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/age-full.csv')

    images_age_dir = 'projects/petfinder/petfinder-previous/data/images-age-preprocessed/'

    age_breed_list = []

    for image_name in tqdm(os.listdir(images_age_dir), desc='Images'):

        animal_id = age_meta[age_meta['id'] == image_name]['id'].values[0]
        animal_age = age_meta[age_meta['id'] == image_name]['age'].values[0]

        animal_breed = breeds_full_meta[breeds_full_meta['id'] == image_name]['breed'].values

        if len(animal_breed) != 0:

            animal_breed = animal_breed[0]

            age_breed_list.append([image_name, animal_age, animal_breed])
        
        else:

            animal_breed = breeds_petfinder_meta[breeds_petfinder_meta['id'] == image_name]['breed'].values

            if len(animal_breed) != 0:

                animal_breed = animal_breed[0]

                animal_breed = animal_breed.replace(' ', ', ')

                age_breed_list.append([image_name, animal_age, animal_breed])
            
            else:

                continue
    
    age_breed_meta = pd.DataFrame(age_breed_list, columns=['id', 'age', 'breed'])
    age_breed_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/age-breed-full.csv', index=False)


def removeCatsDogs():

    d_dir = 'projects/petfinder/petfinder-previous/data/additional-data/dogs-age-preprocessed/'
    d_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/predicted-breeds-dogs.csv')
    b_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/additional-data/breeds-type.csv')

    new_d_breed_list = []

    inc_ids = []
    for index, row in d_meta.iterrows():

        d_id = row['id']
        d_breed = row['breed']
        d_breed = d_breed.replace('\n', '')
        
        d_type = b_meta[b_meta['breed'] == d_breed]['type'].values[0]

        if d_type != '[0, 1]':

            inc_ids.append(d_id)

            shutil.move(d_dir + d_id, 'projects/petfinder/petfinder-previous/data/additional-data/dogs-age-preprocessed-cats/' + d_id)

        else:

            new_d_breed_list.append([d_id, d_breed])

    new_d_breed_meta = pd.DataFrame(new_d_breed_list, columns=['id', 'breed'])
    new_d_breed_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/additional-data/dogs-breeds.csv', index=False)


def preprocessPetfinderAge():

    petfinder_dir = 'projects/petfinder/petfinder-previous/data/images-preprocessed/256/'
    petfinder_age_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/preprocessed_metadata.csv')

    age_list = []

    for image_name in tqdm(os.listdir(petfinder_dir), desc='Process'):

        animal_age = petfinder_age_meta[petfinder_age_meta['id'] == image_name.split('_')[0]]['Age'].values[0]

        animal_type = petfinder_age_meta[petfinder_age_meta['id'] == image_name.split('_')[0]]['Type'].values[0]
        animal_type = evaluateString(animal_type)

        if animal_type != [1, 0]:
            continue

        if animal_age > 3:
            continue

        animal_age = [1, 0, 0, 0]

        age_list.append([image_name, animal_age])

        shutil.copy(petfinder_dir + image_name, 'projects/petfinder/petfinder-previous/data/additional-data/petfinder-wrong-age/' + image_name)

    age_meta = pd.DataFrame(age_list, columns=['id', 'age'])
    age_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/petfinder-age.csv', index=False)


def breedFurSize():

    fur_size_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-fur-size.csv')

    breed_fur_size_list = []

    breed_ids = []

    for _, row in tqdm(fur_size_meta.iterrows(), desc='Progress'):

        breed_id = row['id']
        a_breed = np.zeros(167)
        a_breed[breed_id - 1] = 1
        a_breed = [int(val) for val in a_breed]

        if breed_id in breed_ids:

            continue

        else:

            breed_ids.append(breed_id)

            a_fur = fur_size_meta[fur_size_meta['id'] == breed_id]['FurLength'].values[0]
            a_fur_onehot = [0, 0, 0]
            a_fur_onehot[a_fur - 1] = 1
            a_size = fur_size_meta[fur_size_meta['id'] == breed_id]['MaturitySize'].values[0]
            a_size_onehot = [0, 0, 0, 0]
            a_size_onehot[a_size - 1] = 1

            breed_fur_size_list.append([a_breed, a_fur_onehot, a_size_onehot])

    breed_fur_size_meta = pd.DataFrame(breed_fur_size_list, columns=['breed', 'fur', 'size'])
    breed_fur_size_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/breeds-fur-size-preprocessed.csv', index=False)


def ageFull():

    age_breeds_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/age-breeds.csv')
    fur_size_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/breeds-fur-size-preprocessed.csv')

    full_list = []

    for _, row in age_breeds_meta.iterrows():

        a_id = row['id']
        a_breed = row['breed']
        a_age = row['age']

        a_fur = fur_size_meta[fur_size_meta['breed'] == a_breed]['fur'].values[0]
        a_size = fur_size_meta[fur_size_meta['breed'] == a_breed]['size'].values[0]

        full_list.append([a_id, a_breed, a_fur, a_size, a_age])

    full_meta = pd.DataFrame(full_list, columns=['id', 'breed', 'fur', 'size', 'age'])
    full_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/age-full.csv', index=False)


def petfinderBreedColor():

    petfinder_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/preprocessed_metadata.csv')
    petfinder_breeds_meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-breeds.csv')

    breed_color_list = []

    for _, row in petfinder_breeds_meta.iterrows():

        a_id = row['id']
        a_breed = row['breed']

        a_color = petfinder_meta[petfinder_meta['id'] == a_id.split('_')[0]]['Color'].values[0]

        breed_color_list.append([a_id, a_breed, a_color])

    breed_color_meta = pd.DataFrame(breed_color_list, columns=['id', 'breed', 'color'])
    breed_color_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/petfinder-breed-color.csv', index=False)


def ageTo3classes():

    meta = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/age-full.csv')

    new_meta_list = []

    for _, row in meta.iterrows():

        a_id = row['id']
        a_breed = row['breed']
        a_fur = row['fur']
        a_size = row['size']
        a_age = row['age']

        age_idx = np.argmax(evaluateString(a_age))

        if age_idx == 0:
            new_age = [1, 0, 0]
        elif age_idx == 1:
            new_age = [0, 1, 0]
        else:
            new_age = [0, 0, 1]
        
        new_meta_list.append([a_id, a_breed, a_fur, a_size, new_age])

    new_meta = pd.DataFrame(new_meta_list, columns=['id', 'breed', 'fur', 'size', 'age'])
    new_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/age-full-3classes.csv', index=False)


def petfinderAge3classes():

    petfinder_dir = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed'
    petfinder_prev = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-preprocessed-metadata.csv')

    petfinder_age_list = []

    for image_name in os.listdir(petfinder_dir):

        a_age = petfinder_prev[petfinder_prev['id'] == image_name.split('_')[0]]['Age'].values[0]

        if (a_age <= 6):

            one_hot_age = [1, 0, 0]
        
        elif ((a_age >= 7) and (a_age <= 12)):

            one_hot_age = [0, 1, 0]
        
        else:

            one_hot_age = [0, 0, 1]
        
        petfinder_age_list.append([image_name, one_hot_age])
    
    petfinder_age = pd.DataFrame(petfinder_age_list, columns=['id', 'age'])
    petfinder_age.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/petfinder-age-3classes-preprocessed.csv', index=False)


def petfinderAge4classes():

    petfinder_dir = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed'
    petfinder_prev = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-preprocessed-metadata.csv')

    petfinder_age_list = []

    for image_name in os.listdir(petfinder_dir):

        a_age = petfinder_prev[petfinder_prev['id'] == image_name.split('_')[0]]['Age'].values[0]

        if (a_age <= 6):

            one_hot_age = [1, 0, 0, 0]
        
        elif ((a_age >= 7) and (a_age <= 12)):

            one_hot_age = [0, 1, 0, 0]

        elif ((a_age >= 13) and (a_age < 132)):

            one_hot_age = [0, 0, 1, 0]
        
        else:

            one_hot_age = [0, 0, 0, 1]
        
        petfinder_age_list.append([image_name, one_hot_age])
    
    petfinder_age = pd.DataFrame(petfinder_age_list, columns=['id', 'age'])
    petfinder_age.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/petfinder-age-4classes-preprocessed.csv', index=False)



def createNewPetfinder():

    petfinder_dir = 'projects/petfinder/petfinder-previous/data/petfinder-images-preprocessed'
    petfinder_prev = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/petfinder-preprocessed-metadata.csv')
    full_new = pd.read_csv('projects/petfinder/petfinder-previous/data/metadata/age-3classes-full-last-predicted-breeds.csv')

    petfinder_list = []

    for image_name in os.listdir(petfinder_dir):

        a_color = petfinder_prev[petfinder_prev['id'] == image_name.split('_')[0]]['Color'].values[0]
        a_breed = full_new[full_new['id'] == image_name]['breed'].values[0]
        a_type = full_new[full_new['breed'] == a_breed]['type'].values[0]
        a_age = full_new[full_new['id'] == image_name]['age'].values[0]
        a_fur = full_new[full_new['breed'] == a_breed]['fur'].values[0]
        a_size = full_new[full_new['breed'] == a_breed]['size'].values[0]

        a_speed_idx = petfinder_prev[petfinder_prev['id'] == image_name.split('_')[0]]['AdoptionSpeed'].values[0]
        a_speed = [0, 0, 0, 0, 0]
        a_speed[a_speed_idx] = 1

        petfinder_list.append([image_name, a_type, a_age, a_color, a_breed, a_fur, a_size, a_speed])

    petfinder_meta = pd.DataFrame(petfinder_list, columns=['id', 'type', 'age', 'color', 'breed', 'fur', 'size', 'speed'])
    petfinder_meta.to_csv(path_or_buf='projects/petfinder/petfinder-previous/data/metadata/petfinder-last-speed.csv', index=False)



