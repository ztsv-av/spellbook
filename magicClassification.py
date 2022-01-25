from globalVariables import (
    NUM_EPOCHS, START_EPOCH, BATCH_SIZES, 
    INPUT_SHAPE,
    LOAD_FEATURES, NUM_FEATURES, CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, 
    MODEL_POOLING, DROP_CONNECT_RATE, INITIAL_DROPOUT, DO_BATCH_NORM, FC_LAYERS, DROPOUT_RATES,      
    DO_PREDICTIONS, NUM_CLASSES, OUTPUT_ACTIVATION,
    UNFREEZE, UNFREEZE_FULL, NUM_UNFREEZE_LAYERS,
    LOAD_WEIGHTS, LOAD_MODEL,
    BUILD_AUTOENCODER, 
    DATA_FILEPATHS, TRAIN_FILEPATHS, VAL_FILEPATHS, DO_VALIDATION, MAX_FILES_PER_PART, SHUFFLE_BUFFER_SIZE, RANDOM_STATE,
    METADATA, ID_COLUMN, FEATURE_COLUMN, FULL_RECORD,
    DO_KFOLD, NUM_FOLDS, 
    CLASSIFICATION_CHECKPOINT_PATH, TRAINED_MODELS_PATH, 
    SAVE_TRAIN_WEIGHTS_DIR, SAVE_TRAIN_INFO_DIR,
    LEARNING_RATE,
    LR_EXP, LR_DECAY_STEPS, LR_DECAY_RATE,
    LR_LADDER, LR_LADDER_STEP, LR_LADDER_EPOCHS,  
    PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS,
    FROM_LOGITS, LABEL_SMOOTHING, LOSS_REDUCTION, 
    ACCURACY_THRESHOLD,
    BUILD_AUTOENCODER, DENSE_NEURONS_DATA_FEATURES, DENSE_NEURONS_ENCODER, DENSE_NEURONS_BOTTLE, DENSE_NEURONS_DECODER)

from models import MODELS_CLASSIFICATION, unfreezeModel, buildClassificationImageNetModel, buildDenoisingAutoencoder
from train import classificationCustomTrain
from preprocessFunctions import kerasNormalize
from losses import rootMeanSquaredErrorLossWrapper
from helpers import getFullPaths

import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy(
    devices=["GPU:0", "GPU:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


def classificationCustom():
    """
    XXX
    """

    data_paths_list = np.array(getFullPaths(DATA_FILEPATHS))
    data_paths_list_shuffled = shuffle(data_paths_list)

    for model_name, model_imagenet in MODELS_CLASSIFICATION.items():

        batch_size_per_replica = BATCH_SIZES[model_name]
        batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

        if DO_KFOLD:

            kfold = KFold(NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

            for fold, (train_ix, val_ix) in enumerate(kfold.split(data_paths_list_shuffled)):

                train_paths_list = np.ndarray.tolist(data_paths_list_shuffled[train_ix])
                val_paths_list = np.ndarray.tolist(data_paths_list_shuffled[val_ix])

                max_fileparts_train = len(train_paths_list) // MAX_FILES_PER_PART
                max_fileparts_val = len(val_paths_list) // MAX_FILES_PER_PART

                with strategy.scope():

                    # create model, loss, optimizer and metrics instances here
                    # reset model, optimizer (AND learning rate), loss and metrics after each iteration

                    if BUILD_AUTOENCODER:

                        input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data_layer')

                        input_layers_classification = [input_data_layer]

                        input_features_layers = []

                        if LOAD_FEATURES:

                            for idx, features in enumerate(NUM_FEATURES):

                                input_features_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[features], ), name=('input_features_layer_' + idx))

                                input_features_layers.append(input_features_layer)
                        
                        predictions_features = NUM_FEATURES
                        
                        input_den_autoenc_layers = input_layers_classification + input_features_layers

                        model = buildDenoisingAutoencoder(
                            input_layers, 
                            model_name, model_imagenet,
                            MODEL_POOLING, DROP_CONNECT_RATE, DO_BATCH_NORM, INITIAL_DROPOUT,
                            CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, 
                            FC_LAYERS, DROPOUT_RATES,
                            NUM_CLASSES, OUTPUT_ACTIVATION, 
                            DO_PREDICTIONS,
                            input_features_layers,
                            DENSE_NEURONS_DATA_FEATURES, DENSE_NEURONS_ENCODER, DENSE_NEURONS_BOTTLE, DENSE_NEURONS_DECODER,
                            predictions_features,
                            input_den_autoenc_layers)
                    
                    else:

                        input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data_layer')

                        input_layers = [input_data_layer]

                        if LOAD_FEATURES:

                            for idx, features in enumerate(NUM_FEATURES):

                                input_features_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[features], ), name=('input_features_layer_' + idx))

                                input_layers.append(input_features_layer)

                        model = buildClassificationImageNetModel(
                            input_layers, 
                            model_name, model_imagenet,
                            MODEL_POOLING, DROP_CONNECT_RATE, DO_BATCH_NORM, INITIAL_DROPOUT, 
                            CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, 
                            FC_LAYERS, DROPOUT_RATES,
                            NUM_CLASSES, OUTPUT_ACTIVATION, 
                            DO_PREDICTIONS)

                        if UNFREEZE:

                            num_layers = 0
                            for layer in model.layers:
                                num_layers += 1

                            if UNFREEZE_FULL:

                                to_unfreeze = num_layers

                            else:
                                
                                if ((model_name == 'VGG16') or (model_name == 'VGG19')):

                                    to_unfreeze = num_layers

                                else:

                                    to_unfreeze = NUM_UNFREEZE_LAYERS

                            model = unfreezeModel(model, len(input_layers), DO_BATCH_NORM, to_unfreeze)

                        if LOAD_WEIGHTS:

                            model.load_weights(CLASSIFICATION_CHECKPOINT_PATH)

                    loss_object = tf.losses.CategoricalCrossentropy(
                        from_logits=FROM_LOGITS, label_smoothing=LABEL_SMOOTHING ,reduction=tf.keras.losses.Reduction.NONE)

                    def compute_total_loss(labels, predictions):
                        per_gpu_loss = loss_object(labels, predictions)
                        return tf.nn.compute_average_loss(
                            per_gpu_loss, global_batch_size=batch_size)

                    val_loss = tf.keras.metrics.Mean(name='val_loss')

                    if LR_EXP:
                        exp_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=LEARNING_RATE, decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE)
                        optimizer = tf.keras.optimizers.Adam(learning_rate=exp_learning_rate)

                    else:
                        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

                    train_accuracy = tf.keras.metrics.CategoricalAccuracy(
                        name='train_accuracy')
                    val_accuracy = tf.keras.metrics.CategoricalAccuracy(
                        name='val_accuracy')

                    # rename optimizer weights to train multiple models
                    with K.name_scope(optimizer.__class__.__name__):
                        for i, var in enumerate(optimizer.weights):
                            name = 'variable{}'.format(i)
                            optimizer.weights[i] = tf.Variable(
                                var, name=name)

                normalization_function = kerasNormalize(model_name)

                classificationCustomTrain(
                    batch_size, NUM_EPOCHS, START_EPOCH,
                    train_paths_list, val_paths_list, DO_VALIDATION, fold, max_fileparts_train, max_fileparts_val,
                    METADATA, ID_COLUMN, FEATURE_COLUMN, FULL_RECORD,
                    SHUFFLE_BUFFER_SIZE, PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS, normalization_function,
                    model_name, model,
                    loss_object, val_loss, compute_total_loss,
                    LR_LADDER, LR_LADDER_STEP, LR_LADDER_EPOCHS, optimizer,
                    train_accuracy, val_accuracy,
                    SAVE_TRAIN_INFO_DIR, SAVE_TRAIN_WEIGHTS_DIR,
                    strategy,
                    is_autoencoder=False, pretrained=False)

                print('________________________________________')
                print('\n')
                print('TRAINING FINISHED FOR ' + model_name + ', fold ' + str(fold + 1) + '/' + str(NUM_FOLDS) + ' !')
                print('\n')
                print('________________________________________')

                del train_paths_list
                del val_paths_list
                del max_fileparts_train
                del max_fileparts_val
                del model
                del loss_object
                del val_loss
                del optimizer
                del train_accuracy
                del val_accuracy

                K.clear_session()

                # sleep 120 seconds
                print('Sleeping 120 seconds after training ' + model_name + ', fold ' + str(fold + 1) + '/' + str(NUM_FOLDS) + '. Zzz...')
                time.sleep(120)

        else:

            if DO_VALIDATION:

                train_paths_list = getFullPaths(TRAIN_FILEPATHS)
                max_fileparts_train = len(train_paths_list) // MAX_FILES_PER_PART
                if max_fileparts_train == 0:
                    max_fileparts_train = 1

                val_paths_list = getFullPaths(VAL_FILEPATHS)
                max_fileparts_val = len(val_paths_list) // MAX_FILES_PER_PART
                if max_fileparts_val == 0:
                    max_fileparts_val = 1

            else:

                train_paths_list = getFullPaths(DATA_FILEPATHS)
                max_fileparts_train = len(train_paths_list) // MAX_FILES_PER_PART
                if max_fileparts_train == 0:
                    max_fileparts_train = 1
                    
                val_paths_list = None
                max_fileparts_val = None

            with strategy.scope():

                # create model, loss, optimizer and metrics instances here
                # reset model, optimizer (AND learning rate), loss and metrics after each iteration

                if BUILD_AUTOENCODER:

                    input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data_layer')

                    input_layers_classification = [input_data_layer]

                    input_features_layers = []

                    if LOAD_FEATURES:

                        for idx, features in enumerate(NUM_FEATURES):

                            input_features_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[features], ), name=('input_features_layer_' + idx))

                            input_features_layers.append(input_features_layer)
                    
                    predictions_features = NUM_FEATURES
                    
                    input_den_autoenc_layers = input_layers_classification + input_features_layers

                    model = buildDenoisingAutoencoder(
                        input_layers, 
                        model_name, model_imagenet,
                        MODEL_POOLING, DROP_CONNECT_RATE, DO_BATCH_NORM, INITIAL_DROPOUT,
                        CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, 
                        FC_LAYERS, DROPOUT_RATES,
                        NUM_CLASSES, OUTPUT_ACTIVATION, 
                        DO_PREDICTIONS,
                        input_features_layers,
                        DENSE_NEURONS_DATA_FEATURES, DENSE_NEURONS_ENCODER, DENSE_NEURONS_BOTTLE, DENSE_NEURONS_DECODER,
                        predictions_features,
                        input_den_autoenc_layers)
                
                else:

                    input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data_layer')

                    input_layers = [input_data_layer]

                    if LOAD_FEATURES:

                        for idx, features in enumerate(NUM_FEATURES):

                            input_features_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[features], ), name=('input_features_layer_' + idx))

                            input_layers.append(input_features_layer)

                    model = buildClassificationImageNetModel(
                        input_layers, 
                        model_name, model_imagenet,
                        MODEL_POOLING, DROP_CONNECT_RATE, DO_BATCH_NORM, INITIAL_DROPOUT, 
                        CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, 
                        FC_LAYERS, DROPOUT_RATES,
                        NUM_CLASSES, OUTPUT_ACTIVATION, 
                        DO_PREDICTIONS)

                    if UNFREEZE:

                        num_layers = 0
                        for layer in model.layers:
                            num_layers += 1

                        if UNFREEZE_FULL:

                            to_unfreeze = num_layers

                        else:
                            
                            if ((model_name == 'VGG16') or (model_name == 'VGG19')):

                                to_unfreeze = num_layers

                            else:

                                to_unfreeze = NUM_UNFREEZE_LAYERS

                        model = unfreezeModel(model, len(input_layers), DO_BATCH_NORM, to_unfreeze)

                    if LOAD_WEIGHTS:

                        model.load_weights(CLASSIFICATION_CHECKPOINT_PATH)

                loss_object = rootMeanSquaredErrorLossWrapper(reduction=LOSS_REDUCTION)

                def compute_total_loss(labels, predictions):
                    per_gpu_loss = loss_object(labels, predictions)
                    return tf.nn.compute_average_loss(
                        per_gpu_loss, global_batch_size=batch_size)

                if DO_VALIDATION:
                    val_loss = tf.keras.metrics.Mean(name='val_loss')
                else:
                    val_loss = None

                if LR_EXP:
                    exp_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=LEARNING_RATE, decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=exp_learning_rate)

                else:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

                train_accuracy = tf.keras.metrics.MeanSquaredError(
                    name='train_MSE')
                if DO_VALIDATION:
                    val_accuracy = tf.keras.metrics.MeanSquaredError(
                        name='val_MSE')
                else:
                    val_accuracy = None

                # rename optimizer weights to train multiple models
                with K.name_scope(optimizer.__class__.__name__):
                    for i, var in enumerate(optimizer.weights):
                        name = 'variable{}'.format(i)
                        optimizer.weights[i] = tf.Variable(
                            var, name=name)

            normalization_function = kerasNormalize(model_name)

            classificationCustomTrain(
                batch_size, NUM_EPOCHS, START_EPOCH,
                train_paths_list, val_paths_list, DO_VALIDATION, None, max_fileparts_train, max_fileparts_val,
                METADATA, ID_COLUMN, FEATURE_COLUMN, FULL_RECORD,
                SHUFFLE_BUFFER_SIZE, PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS, normalization_function,
                model_name, model,
                loss_object, val_loss, compute_total_loss,
                LR_LADDER, LR_LADDER_STEP, LR_LADDER_EPOCHS, optimizer,
                train_accuracy, val_accuracy,
                SAVE_TRAIN_INFO_DIR, SAVE_TRAIN_WEIGHTS_DIR,
                strategy,
                is_autoencoder=False, pretrained=False)

            print('________________________________________')
            print('\n')
            print('TRAINING FINISHED FOR ' + model_name + '!')
            print('\n')
            print('________________________________________')

            del train_paths_list
            del val_paths_list
            del max_fileparts_train
            del max_fileparts_val
            del model
            del loss_object
            del val_loss
            del optimizer
            del train_accuracy
            del val_accuracy

            K.clear_session()

            # sleep 120 seconds
            print('Sleeping 120 seconds after training ' + model_name + '. Zzz...')
            time.sleep(120)

        del batch_size_per_replica
        del batch_size
