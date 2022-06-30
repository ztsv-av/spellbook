from globalVariables import (
    NUM_EPOCHS, START_EPOCH, BATCH_SIZES, 
    INPUT_SHAPE,
    USE_TFIMM_MODELS,
    IMAGENET_WEIGHTS, EFFNET_WEIGHTS,
    LOAD_FEATURES, NUM_ADD_CLASSES, CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, 
    MODEL_POOLING, DROP_CONNECT_RATE, INITIAL_DROPOUT, DO_BATCH_NORM, FC_LAYERS, DROPOUT_RATES,      
    DO_PREDICTIONS, NUM_CLASSES, OUTPUT_ACTIVATION,
    UNFREEZE, UNFREEZE_FULL, UNFREEZE_PERCENT, UNFREEZE_BATCHNORM, NUM_UNFREEZE_LAYERS,
    LOAD_WEIGHTS, LOAD_MODEL,
    BUILD_AUTOENCODER, 
    DATA_FILEPATHS, TRAIN_FILEPATHS, VAL_FILEPATHS, DO_VALIDATION, VAL_SPLIT, MAX_FILES_PER_PART, RANDOM_STATE,
    METADATA, ID_COLUMN, TARGET_FEATURE_COLUMNS, ADD_FEATURES_COLUMNS, 
    FILENAME_UNDERSCORE, CREATE_ONEHOT, CREATE_SPARSE, LABEL_IDX, LABEL_IDXS_ADD,     
    DO_KFOLD, NUM_FOLDS, 
    CLASSIFICATION_CHECKPOINT_PATH, TRAINED_MODELS_PATH, 
    SAVE_TRAIN_WEIGHTS_DIR, SAVE_TRAIN_INFO_DIR,
    ARCMARGIN_S, ARCMARGIN_M,
    OPTIMIZER, LEARNING_RATE, MOMENTUM_VALUE, NESTEROV,
    CUSTOM_LRS_EPOCHS,
    LR_EXP, LR_DECAY_STEPS, LR_DECAY_RATE,
    LR_CUSTOM_DECAY, LR_START_DECAY, LR_MAX_DECAY, LR_MIN_DECAY, LR_RAMP_EP_DECAY, LR_SUS_EP_DECAY, LR_VALUE_DECAY,
    LR_LADDER, LR_LADDER_STEP, LR_LADDER_EPOCHS,
    REDUCE_LR_PLATEAU, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MINIMAL_LR, REDUCE_LR_METRIC,   
    PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS,
    FROM_LOGITS, LABEL_SMOOTHING, LOSS_REDUCTION, 
    METRIC_TYPE, ACCURACY_THRESHOLD, F1_SCORE_AVERAGE, F1_SCORE_THRESHOLD,
    BUILD_AUTOENCODER, DENSE_NEURONS_DATA_FEATURES, DENSE_NEURONS_ENCODER, DENSE_NEURONS_BOTTLE, DENSE_NEURONS_DECODER)

from models import MODELS_CLASSIFICATION, TFIMM_MODELS, unfreezeModel, buildClassificationImageNetModel, buildDenoisingAutoencoder
from layers import ArcMarginProduct
from train import classificationCustomTrain
from preprocessFunctions import kerasNormalize
from losses import categoricalFocalLossWrapper, binaryFocalLossWrapper
from optimizers import getLRCallback
from metrics import map5Wrapper, f1Wrapper
from helpers import getFullPaths

import time
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.backend as K
import tfimm
import keras
import efficientnet.keras as efn 
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy(
    devices=["GPU:0", "GPU:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


def classificationCustom():
    """
    main working function that starts the whole training process for classification or encoding-decoding tasks
    it does the following:
        - creates a list with paths to training files
        - iterates through every model initialized in models.py
        - if DO_KFOLD = True it iterates through every fold for each model
        - initializes as much input layers as needed, as well as additional input feature layers
        - creates a model:
            - if BUILD_AUTOENCODER = True it creates an autoencoder with ImageNet model body instead of a modified ImageNet model
            - otherwise it either creates a modified ImageNet model (read more in description of buildClassificationImageNetModel function in models.py file)
            - or any custom model initialized in models.py file
        - if UNFREEZE = True it unfrezees either all layers or a specific number of top layers in the model (transfer learning)
        - optionally loads model weights from a training checkpoint or a whole pretrained model
        - initializes loss function, learning rate, optimizer and metrics
        - if DO_VALIDATION = True it will check a model perfomance after every training epoch
        - calls a function that starts a training process and passes all initialized variables to it
    every action is based on the global variables initialized in globalVariables.py
    """

    data_paths_list = np.array(getFullPaths(DATA_FILEPATHS))
    data_paths_list_shuffled = shuffle(data_paths_list, random_state=RANDOM_STATE)

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

                            for idx, features in enumerate(NUM_ADD_CLASSES):

                                input_features_layer = tf.keras.layers.Input(shape=(), name=('input_features_layer_' + str(idx)))

                                input_features_layers.append(input_features_layer)
                        
                        predictions_features = NUM_ADD_CLASSES
                        
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
                        
                        normalization_function = kerasNormalize(model_name)

                    elif USE_TFIMM_MODELS:

                        input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data_layer')
                        input_features_layer = tf.keras.layers.Input(shape=(NUM_ADD_CLASSES[0], ), name=('input_features_layer_1'))

                        backbone = tfimm.create_model(model_name, nb_classes=0, pretrained="timm")
                        backbone_input = backbone(input_data_layer)

                        classifier_concat = tf.keras.layers.Concatenate()([input_features_layer, backbone_input])
                        classifier_dense = tf.keras.layers.Dense(units=FC_LAYERS[0], activation='relu')(classifier_concat)
                        classifier_prediction = tf.keras.layers.Dense(units=NUM_CLASSES, activation=OUTPUT_ACTIVATION)(classifier_dense)

                        model = tf.keras.Model(inputs=[input_data_layer, input_features_layer], outputs=classifier_prediction)

                        # model = tfimm.create_model(model_name, nb_classes=NUM_CLASSES, pretrained="timm")

                        # model = load_model(CLASSIFICATION_CHECKPOINT_PATH)
                        
                        normalization_function = tfimm.create_preprocessing(model_name, dtype="float32")  

                    else:
                        
                        # model_name = 'EfficientNetB5'
                        # model_imagenet = tf.keras.applications.efficientnet.EfficientNetB5  

                        input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data_layer')

                        input_layers = [input_data_layer]

                        if LOAD_FEATURES:

                            for idx, features in enumerate(NUM_ADD_CLASSES):

                                input_features_layer = tf.keras.layers.Input(shape=(), name=('input_features_layer_' + str(idx)))

                                input_layers.append(input_features_layer)
                        
                        if LOAD_MODEL:

                            model = load_model(CLASSIFICATION_CHECKPOINT_PATH)
                        
                        else:

                            model = buildClassificationImageNetModel(
                                input_layers, 
                                model_name, model_imagenet, IMAGENET_WEIGHTS,
                                MODEL_POOLING, DROP_CONNECT_RATE, DO_BATCH_NORM, INITIAL_DROPOUT, 
                                CONCAT_FEATURES_BEFORE, CONCAT_FEATURES_AFTER, 
                                FC_LAYERS, DROPOUT_RATES,
                                NUM_CLASSES, OUTPUT_ACTIVATION, 
                                DO_PREDICTIONS)

                        # model = efn.EfficientNetB5(
                        #     weights='noisy-student', include_top=False, input_shape=[INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
                        # model.layers[0]._name = 'inp1'
                        # model.trainable = False
                        # input_layers = [1]

                        if UNFREEZE:

                            num_layers = 0
                            for layer in model.layers:
                                num_layers += 1

                            if UNFREEZE_FULL:
                                to_unfreeze = num_layers

                            if UNFREEZE_PERCENT is not None:
                                unfreeze_layers = UNFREEZE_PERCENT / 100
                                to_unfreeze = num_layers[:-unfreeze_layers]

                            else:
                                
                                if ((model_name == 'VGG16') or (model_name == 'VGG19')):

                                    to_unfreeze = num_layers

                                else:

                                    to_unfreeze = NUM_UNFREEZE_LAYERS[model_name]

                            model = unfreezeModel(model, len(input_layers), DO_BATCH_NORM, to_unfreeze, UNFREEZE_BATCHNORM)

                        if LOAD_WEIGHTS:

                            model.load_weights(CLASSIFICATION_CHECKPOINT_PATH)

                        gap1 = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
                        gap2 = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-5].output)
                        gap3 = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-9].output)
                        gap4 = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-14].output)

                        embedding =  tf.concat([gap1, gap2, gap3, gap4], axis=1)

                        embedding = tf.keras.layers.Dropout(0.3)(embedding)
                        embedding = tf.keras.layers.Dense(2048)(embedding)

                        # def dropoutDense(x, drop, dense):
                        #     x = tf.keras.layers.Dropout(drop)(x)
                        #     return tf.keras.layers.Dense(dense)(x)

                        # x1 = dropoutDense(gap1, drop=0.3, dense=2560)
                        # x2 = dropoutDense(gap2, drop=0.3, dense=640)
                        # x3 = dropoutDense(gap3, drop=0.3, dense=640)
                        # x4 = dropoutDense(gap4, drop=0.3, dense=3840)

                        # embedding =  tf.concat([x1,x2,x3,x4], axis=1)

                        input_features_layer = tf.keras.layers.Input(shape=(), name=('input_features_layer_1'))

                        arc_margin = ArcMarginProduct(
                            n_classes=NUM_CLASSES, s=ARCMARGIN_S, m=ARCMARGIN_M, name='head/arcmargin', dtype='float32')
                        arc_margin_layer = arc_margin([embedding, input_features_layer])

                        output = tf.keras.layers.Softmax(dtype='float32')(arc_margin_layer)

                        model_arc = tf.keras.models.Model(inputs=[input_data_layer, input_features_layer], outputs=[output])

                        normalization_function = kerasNormalize(model_name)

                    # loss_object = categoricalFocalLossWrapper(reduction=LOSS_REDUCTION)

                    loss_object = tf.losses.SparseCategoricalCrossentropy(
                        from_logits=FROM_LOGITS, name='train_SCC_loss', reduction=tf.keras.losses.Reduction.NONE)

                    def compute_total_loss(labels, predictions):
                        per_gpu_loss = loss_object(labels, predictions)
                        return tf.nn.compute_average_loss(
                            per_gpu_loss, global_batch_size=batch_size)

                    val_loss = tf.keras.metrics.Mean(name='val_SCC_mean_loss')

                    if LR_EXP:

                        exp_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=LEARNING_RATE, decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE)
                        optimizer = tf.keras.optimizers.Adam(learning_rate=exp_learning_rate)

                    elif LR_CUSTOM_DECAY:

                        RESUME_TRAINING = True if (START_EPOCH != 0) else False
                        # schedule = getLRCallback(
                        #     LR_START_DECAY, LR_MAX_DECAY, LR_MIN_DECAY, LR_RAMP_EP_DECAY, LR_SUS_EP_DECAY, LR_VALUE_DECAY,
                        #     batch_size, epoch)
                        decay_learning_rate = tf.keras.optimizers.schedules.LearningRateSchedule(getLRCallback)
                        optimizer = tf.keras.optimizers.Adam(learning_rate=decay_learning_rate)

                    else:

                        if OPTIMIZER == 'Adam':
                            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
                        
                        elif OPTIMIZER == 'SGD':
                            optimizer = tf.keras.optimizers.SGD(
                                learning_rate=LEARNING_RATE, momentum=MOMENTUM_VALUE, nesterov=NESTEROV)

                    train_metric_1 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_SCA_metric')
                    train_metric_2 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='train_STOPKCA_metric')

                    train_metrics = [train_metric_1, train_metric_2]

                    val_metric_1 = tf.keras.metrics.SparseCategoricalAccuracy(name='val_SCA_metric')
                    val_metric_2 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='val_STOPKCA_metric')

                    val_metrics = [val_metric_1, val_metric_2]

                    # train_metric = map5Wrapper()
                    # val_metric = map5Wrapper()

                    # rename optimizer weights to train multiple models
                    with K.name_scope(optimizer.__class__.__name__):
                        for i, var in enumerate(optimizer.weights):
                            name = 'variable{}'.format(i)
                            optimizer.weights[i] = tf.Variable(
                                var, name=name)

                classificationCustomTrain(
                    NUM_EPOCHS, START_EPOCH, batch_size, NUM_CLASSES, NUM_ADD_CLASSES,  
                    train_paths_list, val_paths_list, DO_VALIDATION, max_fileparts_train, max_fileparts_val, fold,
                    METADATA, ID_COLUMN, TARGET_FEATURE_COLUMNS, ADD_FEATURES_COLUMNS, 
                    FILENAME_UNDERSCORE, CREATE_ONEHOT, CREATE_SPARSE, LABEL_IDX, LABEL_IDXS_ADD, 
                    PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS, normalization_function,
                    model_name, model_arc,
                    loss_object, val_loss, compute_total_loss,
                    CUSTOM_LRS_EPOCHS,
                    LR_LADDER, LR_LADDER_STEP, LR_LADDER_EPOCHS, 
                    REDUCE_LR_PLATEAU, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MINIMAL_LR, REDUCE_LR_METRIC,
                    optimizer,
                    METRIC_TYPE, train_metrics, val_metrics,
                    SAVE_TRAIN_INFO_DIR, SAVE_TRAIN_WEIGHTS_DIR,
                    strategy)

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
                del model_arc
                del loss_object
                del val_loss
                del optimizer
                del train_metrics
                del val_metrics

                K.clear_session()

                # sleep 120 seconds
                print('Sleeping 120 seconds after training ' + model_name + ', fold ' + str(fold + 1) + '/' + str(NUM_FOLDS) + '. Zzz...')
                time.sleep(120)

        else:

            train_paths_list = getFullPaths(TRAIN_FILEPATHS)
            max_fileparts_train = len(train_paths_list) // MAX_FILES_PER_PART
            if max_fileparts_train == 0:
                max_fileparts_train = 1

            if DO_VALIDATION:

                val_paths_list = getFullPaths(VAL_FILEPATHS)
                max_fileparts_val = len(val_paths_list) // MAX_FILES_PER_PART
                if max_fileparts_val == 0:
                    max_fileparts_val = 1

            else:
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

                        for idx, features in enumerate(NUM_ADD_CLASSES):

                            input_features_layer = tf.keras.layers.Input(shape=(features, ), name=('input_features_layer_' + str(idx)))

                            input_features_layers.append(input_features_layer)
                    
                    predictions_features = NUM_ADD_CLASSES
                    
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
                
                elif model_name in TFIMM_MODELS and USE_TFIMM_MODELS:

                    # input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data_layer')
                    # input_features_layer = tf.keras.layers.Input(shape=(NUM_ADD_CLASSES[0], ), name=('input_features_layer_1'))

                    # backbone = tfimm.create_model(model_name, nb_classes=0, pretrained="timm")
                    # backbone_input = backbone(input_data_layer)
                    # classifier_concat = tf.keras.layers.Concatenate()([input_features_layer, backbone_input])
                    # classifier_dense = tf.keras.layers.Dense(units=NUM_CLASSES, activation=OUTPUT_ACTIVATION)(classifier_concat)

                    # model = tf.keras.Model(inputs=[input_data_layer, input_features_layer], outputs=classifier_dense)

                    input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], ), name='input_data_layer')

                    model = tfimm.create_model(model_name, nb_classes=0, pretrained="timm")
                    backbone = model(input_data_layer)

                    dropout = tf.keras.layers.Dropout(0.2)(backbone)
                    output = tf.keras.layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION)(dropout)

                    model_gaps = tf.keras.Model(inputs=[input_data_layer], outputs=[output])
                    model_gaps.trainable = True

                    # model = 'None'
                    # model_gaps = load_model(CLASSIFICATION_CHECKPOINT_PATH)
                    # model_gaps.trainable = True
                    # LEARNING_RATE = 0.000125
                    # START_EPOCH = 20

                    normalization_function = tfimm.create_preprocessing(model_name, dtype="float32")

                    LEARNING_RATE = 0.001
                    START_EPOCH = 0

                else:

                    if model_name == 'EfficientNetB#': # load model

                        model = 'None'
                        model_gaps = load_model(CLASSIFICATION_CHECKPOINT_PATH)
                        model_gaps.trainable = True
                        LEARNING_RATE = 0.0000625
                        START_EPOCH = 20
                    
                    else:

                        LEARNING_RATE = 0.001
                        START_EPOCH = 0

                        # model_name = 'Xception'
                        # model_imagenet = tf.keras.applications.Xception

                        input_data_layer = tf.keras.layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[ 2], ), name='input_data_layer')
                        input_layers = [input_data_layer]

                        if LOAD_FEATURES:
                            for idx, features in enumerate(NUM_ADD_CLASSES):
                                input_features_layer = tf.keras.layers.Input(shape=(), name=('input_features_layer_' + str(idx)))
                                input_layers.append(input_features_layer)
                        
                        if LOAD_MODEL:
                            model = load_model(CLASSIFICATION_CHECKPOINT_PATH)
                        
                        else:
                            
                            if model_name  == 'EfficientNetB5':

                                model = efn.EfficientNetB5(
                                    weights=EFFNET_WEIGHTS, include_top=False, input_shape=[INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
                                model.layers[0]._name = 'inp1'
                                model.trainable = True
                            
                            else:

                                model = buildClassificationImageNetModel(
                                    input_layers, 
                                    model_name, model_imagenet, IMAGENET_WEIGHTS,
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
                            
                            elif UNFREEZE_PERCENT is not None:
                                to_unfreeze = int(num_layers * UNFREEZE_PERCENT / 100)

                            else:
                                
                                if ((model_name == 'VGG16') or (model_name == 'VGG19')):
                                    to_unfreeze = num_layers

                                else:
                                    to_unfreeze = NUM_UNFREEZE_LAYERS[model_name]

                            # model = unfreezeModel(model, len(input_layers), DO_BATCH_NORM, to_unfreeze, UNFREEZE_BATCHNORM)
                            for idx in range(-to_unfreeze, 0, 1):
                                layer = model.layers[idx]
                                # if not isinstance(layer, tf.keras.layers.BatchNormalization):
                                layer.trainable = True

                        if LOAD_WEIGHTS:
                            model.load_weights(CLASSIFICATION_CHECKPOINT_PATH)

                        gap1=tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
                        # gap2=tf.keras.layers.GlobalAveragePooling2D()(model.layers[-4].output)
                        # gap3=tf.keras.layers.GlobalAveragePooling2D()(model.layers[-8].output)
                        # gap4=tf.keras.layers.GlobalAveragePooling2D()(model.layers[-11].output)

                        # embedding =  tf.keras.layers.Concatenate(axis=1)([gap1])
                        embedding = tf.keras.layers.Dropout(0.2)(gap1)
                        # embedding = tf.keras.layers.Dense(512)(embedding)

                        output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION)(embedding)

                        model_gaps = tf.keras.models.Model(inputs=model.inputs, outputs=[output_layer])

                        # model_gaps = load_model(CLASSIFICATION_CHECKPOINT_PATH)
                        # for layer in model_gaps.layers:
                        #     if not isinstance(layer, tf.keras.layers.BatchNormalization):
                        #         layer.trainable = True

                    normalization_function = kerasNormalize(model_name)

                loss_object = binaryFocalLossWrapper(reduction=None)

                # loss_object = tf.losses.BinaryCrossentropy(
                #     from_logits=FROM_LOGITS, name='train_BC_loss', reduction=tf.keras.losses.Reduction.NONE)

                def compute_total_loss(labels, predictions):
                    per_gpu_loss = loss_object(labels, predictions)
                    return tf.nn.compute_average_loss(
                        per_gpu_loss, global_batch_size=batch_size)

                if DO_VALIDATION:
                    val_loss = tf.keras.metrics.Mean(name='val_BC_mean_loss')
                else:
                    val_loss = None

                if LR_EXP:
                    exp_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=LEARNING_RATE, decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=exp_learning_rate)

                else:

                    if OPTIMIZER == 'Adam':
                        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
                    
                    elif OPTIMIZER == 'SGD':
                        optimizer = tf.keras.optimizers.SGD(
                            learning_rate=LEARNING_RATE, momentum=MOMENTUM_VALUE, nesterov=NESTEROV)

                train_metric_1 = tf.keras.metrics.BinaryAccuracy(threshold=ACCURACY_THRESHOLD, name='train_BA')
                train_metric_2 = tfa.metrics.F1Score(
                    num_classes=NUM_CLASSES, average=F1_SCORE_AVERAGE, 
                    threshold=F1_SCORE_THRESHOLD, name='train_F1')
                train_metrics = [train_metric_1, train_metric_2]

                if DO_VALIDATION:
                    val_metric_1 = tf.keras.metrics.BinaryAccuracy(threshold=ACCURACY_THRESHOLD, name='val_BA')
                    val_metric_2 = tfa.metrics.F1Score(
                        num_classes=NUM_CLASSES, average=F1_SCORE_AVERAGE, 
                        threshold=F1_SCORE_THRESHOLD, name='val_F1')
                    val_metrics = [val_metric_1, val_metric_2]

                else:
                    val_metrics = None

                # rename optimizer weights to train multiple models
                with K.name_scope(optimizer.__class__.__name__):
                    for i, var in enumerate(optimizer.weights):
                        name = 'variable{}'.format(i)
                        optimizer.weights[i] = tf.Variable(
                            var, name=name)

            classificationCustomTrain(
                NUM_EPOCHS, START_EPOCH, batch_size, NUM_CLASSES, NUM_ADD_CLASSES, 
                train_paths_list, val_paths_list, DO_VALIDATION, max_fileparts_train, max_fileparts_val, None,
                METADATA, ID_COLUMN, TARGET_FEATURE_COLUMNS, ADD_FEATURES_COLUMNS, 
                FILENAME_UNDERSCORE, CREATE_ONEHOT, CREATE_SPARSE, LABEL_IDX, LABEL_IDXS_ADD,  
                PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS, normalization_function,
                model_name, model_gaps,
                loss_object, val_loss, compute_total_loss,
                CUSTOM_LRS_EPOCHS,
                LR_LADDER, LR_LADDER_STEP, LR_LADDER_EPOCHS, 
                REDUCE_LR_PLATEAU, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MINIMAL_LR, REDUCE_LR_METRIC,
                optimizer,
                METRIC_TYPE, train_metrics, val_metrics,
                SAVE_TRAIN_INFO_DIR, SAVE_TRAIN_WEIGHTS_DIR,
                strategy)

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
            del model_gaps
            del loss_object
            del val_loss
            del optimizer
            del train_metrics
            del val_metrics

            K.clear_session()

            # sleep 120 seconds
            print('Sleeping 120 seconds after training ' + model_name + '. Zzz...')
            time.sleep(120)

        del batch_size_per_replica
        del batch_size
