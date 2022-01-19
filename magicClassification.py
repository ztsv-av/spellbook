from globalVariables import (
    BATCH_SIZES, CHECKPOINT_PATH, NUM_EPOCHS, START_EPOCH, NUM_CLASSES, INPUT_SHAPE, DO_PREDICTIONS,
    DATA_FILEPATHS, TRAIN_FILEPATHS, VAL_FILEPATHS, DO_VALIDATION, DO_KFOLD, NUM_FOLDS, RANDOM_STATE, MAX_FILES_PER_PART,
    METADATA, ID_COLUMN, FEATURE_COLUMN, FULL_RECORD,
    SHUFFLE_BUFFER_SIZE ,PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS,
    DO_BATCH_NORM, CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, NUM_FEATURES, FC_LAYERS, INITIAL_DROPOUT, DROPOUT_RATES, DROP_CONNECT_RATE, AUTOENCODER_FC, IMAGE_FEATURE_EXTRACTOR_FC, OUTPUT_ACTIVATION, MODEL_POOLING, UNFREEZE, UNFREEZE_FULL, NUM_UNFREEZE_LAYERS,
    LEARNING_RATE, LR_DECAY_STEPS, LR_DECAY_RATE, LR_LADDER, LR_LADDER_STEP, LR_LADDER_EPOCHS, LR_EXP, FROM_LOGITS, LABEL_SMOOTHING, LOSS_REDUCTION, ACCURACY_THRESHOLD, 
    SAVE_TRAIN_WEIGHTS_DIR, SAVE_TRAIN_INFO_DIR, LOAD_WEIGHTS, LOAD_MODEL, CLASSIFICATION_CHECKPOINT_PATH)

from models import MODELS_CLASSIFICATION, unfreezeModel, userDefinedModel, buildAutoencoderPetfinder, buildClassificationImageNetModel, buildClassificationPretrainedModel
from train import classificationCustomTrain
from preprocessFunctions import kerasNormalize
from losses import rootMeanSquaredErrorLossWrapper, categoricalFocalLossWrapper
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

                    input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data')
                    input_features_type_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[0], ), name='input_features_type')
                    # input_features_age_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[1], ), name='input_features_age')
                    input_features_color_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[1], ), name='input_features_color')
                    # input_features_breed_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[3], ), name='input_features_breed')
                    # input_features_fur_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[4], ), name='input_features_fur')
                    # input_features_size_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[5], ), name='input_features_size')

                    input_layers = [input_data_layer, input_features_type_layer, input_features_color_layer] # , input_features_age_layer, input_features_color_layer, input_features_breed_layer, input_features_fur_layer, input_features_size_layer]

                    classification_model = buildClassificationImageNetModel(
                        input_layers, model_name, model_imagenet,
                        MODEL_POOLING, DROP_CONNECT_RATE, DO_BATCH_NORM, INITIAL_DROPOUT, CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, FC_LAYERS, DROPOUT_RATES,
                        NUM_CLASSES, OUTPUT_ACTIVATION, trainable=False, do_predictions=DO_PREDICTIONS)

                    if UNFREEZE:

                        num_layers = 0
                        for layer in classification_model.layers:
                            num_layers += 1

                        if UNFREEZE_FULL:

                            to_unfreeze = num_layers

                        else:
                            
                            if ((model_name == 'VGG16') or (model_name == 'VGG19')):

                                to_unfreeze = num_layers

                            else:

                                to_unfreeze = NUM_UNFREEZE_LAYERS

                        classification_model = unfreezeModel(classification_model, len(input_layers), DO_BATCH_NORM, to_unfreeze)

                    if LOAD_WEIGHTS:

                        classification_model.load_weights(CLASSIFICATION_CHECKPOINT_PATH)

                    loss_object = tf.losses.CategoricalCrossentropy(
                        from_logits=FROM_LOGITS, label_smoothing=LABEL_SMOOTHING ,reduction=tf.keras.losses.Reduction.NONE)

                    # loss_object = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

                    # loss_object = categoricalFocalLossWrapper(reduction=None)

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

                    # train_accuracy = tf.keras.metrics.RootMeanSquaredError(
                    #     name='train_RMSE')
                    # val_accuracy = tf.keras.metrics.RootMeanSquaredError(
                    #     name='val_RMSE')

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
                    model_name, classification_model,
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
                del classification_model
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

                # if LOAD_MODEL:

                #     classification_model = load_model(CLASSIFICATION_CHECKPOINT_PATH)
                
                # else:

                #     input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data')
                #     input_features_type_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[0], ), name='input_features_type')
                #     input_features_breed_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[1], ), name='input_features_breed')
                #     input_features_fur_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[2], ), name='input_features_fur')
                #     input_features_size_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[3], ), name='input_features_size')
                #     input_features_color_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[4], ), name='input_features_color')
                #     input_features_age_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[5], ), name='input_features_age')

                #     input_layers = [input_data_layer, input_features_type_layer, input_features_breed_layer, input_features_fur_layer, input_features_size_layer, input_features_color_layer, input_features_age_layer] # , input_features_type_layer, input_features_age_layer, input_features_color_layer, input_features_breed_layer, input_features_fur_layer, input_features_size_layer]

                #     classification_model = buildClassificationImageNetModel(
                #         input_layers, model_name, model_imagenet,
                #         MODEL_POOLING, DROP_CONNECT_RATE, DO_BATCH_NORM, INITIAL_DROPOUT, CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, FC_LAYERS, DROPOUT_RATES,
                #         NUM_CLASSES, OUTPUT_ACTIVATION, trainable=False, do_predictions=DO_PREDICTIONS)

                #     if UNFREEZE:

                #         num_layers = 0
                #         for layer in classification_model.layers:
                #             num_layers += 1

                #         if UNFREEZE_FULL:

                #             to_unfreeze = num_layers

                #         else:
                            
                #             if ((model_name == 'VGG16') or (model_name == 'VGG19')):

                #                 to_unfreeze = num_layers

                #             else:

                #                 to_unfreeze = NUM_UNFREEZE_LAYERS

                #         classification_model = unfreezeModel(classification_model, len(input_layers), DO_BATCH_NORM, to_unfreeze)

                #     if LOAD_WEIGHTS:

                #         classification_model.load_weights(CLASSIFICATION_CHECKPOINT_PATH)
                
                # denoising autoencoder

                input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data')
                input_layers_classification = [input_data_layer]
                classification_model = buildClassificationImageNetModel(
                    input_layers_classification, model_name, model_imagenet,
                    MODEL_POOLING, DROP_CONNECT_RATE, False, None, False, False, None, None,
                    NUM_CLASSES, OUTPUT_ACTIVATION, trainable=False, do_predictions=False)

                input_features_type_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[0], ), name='input_features_type')
                input_features_breed_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[1], ), name='input_features_breed')
                input_features_fur_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[2], ), name='input_features_fur')
                input_features_size_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[3], ), name='input_features_size')
                input_features_color_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[4], ), name='input_features_color')
                input_features_age_layer = tf.keras.layers.Input(shape=(NUM_FEATURES[5], ), name='input_features_age')

                features_dense_64 = tf.keras.layers.Dense(64, activation='relu', name="feature_64", use_bias=False)(classification_model.layers[-1].output)

                concat_layer = tf.keras.layers.Concatenate(name='concat_features')([features_dense_64, input_features_type_layer, input_features_breed_layer, input_features_fur_layer, input_features_size_layer, input_features_color_layer, input_features_age_layer])
                
                features_dense_512_enc = tf.keras.layers.Dense(512, activation='relu', name="feature_512_enc", use_bias=False)(concat_layer)
                features_dense_256_bottle = tf.keras.layers.Dense(256, activation='relu', name="feature_256_bottle", use_bias=False)(features_dense_512_enc)
                features_dense_512_dec = tf.keras.layers.Dense(512, activation='relu', name="feature_512_dec", use_bias=False)(features_dense_256_bottle)

                prediction_type = tf.keras.layers.Dense(NUM_FEATURES[0], activation='softmax', name='prediction_type')(features_dense_512_dec)
                prediction_breed = tf.keras.layers.Dense(NUM_FEATURES[1], activation='softmax', name='prediction_breed')(features_dense_512_dec)
                prediction_fur = tf.keras.layers.Dense(NUM_FEATURES[2], activation='softmax', name='prediction_fur')(features_dense_512_dec)
                prediction_size = tf.keras.layers.Dense(NUM_FEATURES[3], activation='softmax', name='prediction_size')(features_dense_512_dec)
                prediction_color = tf.keras.layers.Dense(NUM_FEATURES[4], activation='sigmoid', name='prediction_color')(features_dense_512_dec)
                prediction_age = tf.keras.layers.Dense(NUM_FEATURES[5], activation='softmax', name='prediction_age')(features_dense_512_dec)

                input_layers = [input_data_layer, input_features_type_layer, input_features_breed_layer, input_features_fur_layer, input_features_size_layer, input_features_color_layer, input_features_age_layer]
                prediction_layers = [prediction_type, prediction_breed, prediction_fur, prediction_size, prediction_color, prediction_age]

                denoising_autoencoder = tf.keras.models.Model(inputs=input_layers, outputs=prediction_layers)
                
                # predictions = tf.keras.layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION, name='predictions')(concat_layer)
                # classification_model = tf.keras.models.Model(inputs=[classification_model.input, input_image_features_breed_layer, input_image_features_fur_layer], outputs=predictions)

                # autoencoder = buildAutoencoderPetfinder(
                #     imageFeatureExtractor.output.shape[1], NUM_FEATURES, IMAGE_FEATURE_EXTRACTOR_FC, AUTOENCODER_FC)

                # concat_layer = tf.keras.layers.Concatenate(name='concat_features_dense')([autoencoder.layers[-4].output, autoencoder.layers[-3].output, autoencoder.layers[-2].output, autoencoder.layers[1].output])
                # dense_1 = tf.keras.layers.Dense(256, activation='relu')(concat_layer)
                # dropout_1 = tf.keras.layers.Dropout(0.5)(dense_1)
                # dense_2 = tf.keras.layers.Dense(64, activation='relu')(dropout_1)
                # predictions = tf.keras.layers.Dense(NUM_CLASSES, name='predictions')(dense_2)

                # autoencoder_classification = tf.keras.models.Model(inputs=[autoencoder.layers[0].output, autoencoder.layers[2].output], outputs=predictions)

                # loss_object = categoricalFocalLossWrapper(reduction=LOSS_REDUCTION)

                # loss_object = tf.losses.CategoricalCrossentropy(
                #     from_logits=FROM_LOGITS, label_smoothing=LABEL_SMOOTHING, reduction=tf.keras.losses.Reduction.NONE)

                loss_object = rootMeanSquaredErrorLossWrapper(reduction=LOSS_REDUCTION)

                # loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

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

                # train_accuracy = tf.keras.metrics.CategoricalAccuracy(
                #     name='train_accuracy')
                # if DO_VALIDATION:
                #     val_accuracy = tf.keras.metrics.CategoricalAccuracy(
                #         name='val_accuracy')
                # else:
                #     val_accuracy = None

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
                model_name, denoising_autoencoder,
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
            del denoising_autoencoder
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
