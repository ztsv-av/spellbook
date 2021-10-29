from globalVariables import (
    BATCH_SIZES, NUM_EPOCHS, NUM_CLASSES, INPUT_SHAPE, FC_LAYERS, DROPOUT_RATES, TRAIN_FILEPATHS, VAL_FILEPATHS,
    MAX_FILEPARTS_TRAIN, MAX_FILEPARTS_VAL, PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS, OUTPUT_ACTIVATION, MODEL_POOLING,
    LEARNING_RATE, LR_DECAY_STEPS, LR_DECAY_RATE, SAVE_TRAIN_WEIGHTS_DIR, SAVE_TRAIN_INFO_DIR)

from models import MODELS_CLASSIFICATION, userDefinedModel
from helpers import buildClassificationImageNetModel
from train import classificationCustomTrain
from preprocessFunctions import minMaxNormalizeNumpy
from losses import rootMeanSquaredErrorLoss

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy(
    devices=["GPU:0", "GPU:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


def сlassificationСustom():
    """
    XXX
    """

    for model_name, model_imagenet in MODELS_CLASSIFICATION.items():

        batch_size_per_replica = BATCH_SIZES[model_name]
        batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

        with strategy.scope():

            # create model, loss, optimizer and metrics instances here
            # reset model, optimizer (AND learning rate), loss and metrics after each iteration

            # classification_model = userDefinedModel(NUM_CLASSES, OUTPUT_ACTIVATION)

            classification_model = buildClassificationImageNetModel(
                model_imagenet, INPUT_SHAPE, MODEL_POOLING, FC_LAYERS, DROPOUT_RATES, NUM_CLASSES, OUTPUT_ACTIVATION, trainable=True)

            loss_object = tf.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

            # loss_object = rootMeanSquaredErrorLoss

            def compute_total_loss(labels, predictions):
                per_gpu_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(
                    per_gpu_loss, global_batch_size=batch_size)

            val_loss = tf.keras.metrics.Mean(name='val_loss')

            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=LEARNING_RATE, decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='train_accuracy')
            val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
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

        classificationCustomTrain(
            batch_size, NUM_EPOCHS,
            TRAIN_FILEPATHS, VAL_FILEPATHS, MAX_FILEPARTS_TRAIN, MAX_FILEPARTS_VAL,
            PERMUTATIONS_CLASSIFICATION, DO_PERMUTATIONS, minMaxNormalizeNumpy,
            classification_model,
            loss_object, val_loss, compute_total_loss,
            optimizer,
            train_accuracy, val_accuracy,
            SAVE_TRAIN_INFO_DIR, SAVE_TRAIN_WEIGHTS_DIR,
            model_name, strategy)

        print('________________________________________')
        print('\n')
        print('TRAINING FINISHED FOR ' + model_name + '!')
        print('\n')
        print('________________________________________')

        del batch_size_per_replica
        del batch_size
        del classification_model
        del loss_object
        del val_loss
        del learning_rate
        del optimizer
        del train_accuracy
        del val_accuracy

        K.clear_session()
