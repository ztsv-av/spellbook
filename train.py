# IMPORTS
import tensorflow as tf
import tensorflow.keras.backend as K

import os
from sklearn.utils import shuffle

# VARIABLES & FUNCTIONS
from globalVariables import (NUM_EPOCHS, BATCH_SIZES_256, SAVE_MODELS_DIR, SAVE_TRAINING_CSVS_DIR, 
                            TRAIN_FILES_PATH_DETECTION, CHECKPOINT_SAVE_DIR, REDUCE_LR_PATIENCE, 
                            REDUCE_LR_FACTOR, EARLY_STOPPING_PATIENCE,NUM_EPOCHS_DETECTION, 
                            BATCH_SIZE_DETECTION, TRAIN_META_DETECTION, TRAIN_FILES_PATH, VAL_FILES_PATH)

from prepareTrainDataset import prepareClassificationDataset, prepareDetectionDataset


# def classificationTrain(
#     model_name, model, batch_sizes, num_epochs, preprocessing_dict, strategy, lr_patience, lr_factor, 
#     early_stopping_patience, save_models_dir, save_csvs_dir):
#     """
#     Trains models in sequence stored in 'MODELS_CLASSIFICATION'.
#     For correct training, create optimizer, loss and metrics variables each iteration (in a for loop). 
#     Train and validation data files are shuffled before each model starts training. 
#     For each batch 'GeneratorInstance' is used. Callbacks save the best-fit weights in loss terms (see callbacks.py for more functionality).
#     Deletes all variables and clears session each time one model fully trained.
#     """

#     batch_size_per_replica = batch_sizes[model_name]
#     batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

#     train_dataset, val_dataset, train_len, val_len = prepareClassificationDataset(
#         batch_size, preprocessing_dict)

#     steps_per_epoch_train = int(train_len // batch_size)
#     steps_per_epoch_val = int(val_len // batch_size)

#     # reset callbacks for each model
#     callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=lr_patience, verbose=1, factor=lr_factor),
#                     tf.keras.callbacks.EarlyStopping(
#                         monitor='val_loss', verbose=1, patience=early_stopping_patience),
#                     tf.keras.callbacks.ModelCheckpoint(filepath=save_models_dir + model_name + '/' + model_name +
#                                                     '_epoch-{epoch:02d}.h5', monitor='val_loss', save_best_only=False, verbose=1),
#                     tf.keras.callbacks.CSVLogger(save_csvs_dir + model_name + '_history.csv')]

#     model.fit(x=train_dataset,
#         steps_per_epoch=steps_per_epoch_train,
#         validation_data=val_dataset,
#         validation_steps=steps_per_epoch_val,
#         callbacks=callbacks,
#         epochs=num_epochs,
#         verbose=1)

#     del batch_size_per_replica
#     del batch_size
#     del steps_per_epoch_train
#     del steps_per_epoch_val
#     del train_dataset
#     del val_dataset
#     del train_len
#     del val_len
#     del callbacks


def classificationTrainStep(inputs, compute_total_loss, optimizer, train_accuracy, model):

    images, labels = inputs

    with tf.GradientTape() as tape:

        predictions = model(images, training=True)
        loss = compute_total_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)

    return loss


def classificationDistributedTrainStep(inputs, compute_total_loss, optimizer, train_accuracy, model, strategy):

    per_replica_losses = strategy.run(classificationTrainStep, args=(
        inputs, compute_total_loss, optimizer, train_accuracy, model))
    
    reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    # test if per replica training works
    tf.print(per_replica_losses.values)
    tf.print(reduced_loss)

    return reduced_loss


def classificationValStep(inputs, loss_object, val_loss, val_accuracy, model):

    images, labels = inputs

    predictions = model(images, training=False)
    val_batch_loss = loss_object(labels, predictions)

    val_loss.update_state(val_batch_loss)
    val_accuracy.update_state(labels, predictions)


def classificationDistributedValStep(inputs, loss_object, val_loss, val_accuracy, model, strategy):

    return strategy.run(classificationValStep, args=(inputs, loss_object, val_loss, val_accuracy, model))


def classificationCustomTrain(
    batch_size, num_epochs, train_files_path, val_files_path, permutations, buffer_size, model, loss_object, 
    compute_total_loss, val_loss, optimizer, train_accuracy, val_accuracy, strategy):

    for epoch in range(num_epochs):

        train_distributed_dataset, val_distributed_dataset, _, _ = prepareClassificationDataset(
            batch_size, train_files_path, val_files_path, permutations, buffer_size)

        total_loss = 0.0
        num_batches = 0

        for batch in train_distributed_dataset:

            total_loss += classificationDistributedTrainStep(
                batch, loss_object, compute_total_loss, optimizer, train_accuracy, model, strategy)
            num_batches += 1

        train_loss = total_loss / num_batches

        for batch in val_distributed_dataset:

            classificationDistributedValStep(
                batch, loss_object, val_loss, val_accuracy, model, strategy)

        template = (
            "Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, " "Validation Accuracy: {}")
        print(template.format(epoch+1, train_loss, train_accuracy.result()
                                * 100, val_loss.result(), val_accuracy.result()*100))

        val_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()


def detectionTrainStep(
        image_list, groundtruth_boxes_list, groundtruth_classes_list, model, optimizer, vars_to_fine_tune):
    """
    A single training iteration.

    parameters
    ----------
    image_list: array
        array of [1, height, width, 3] Tensor of type tf.float32.
        images reshaped to model's preprocess function
    groundtruth_boxes_list: array 
        array of Tensors of shape [num_boxes, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
    groundtruth_classes_list: array 
        list of Tensors of shape [num_boxes, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    returns
    -------
    total_loss: scalar tensor
        represents the total loss for the input batch.
    loc_loss: scalar tensor
        represents the localization loss for the input batch.
    class_loss: scalar tensor
        represents the classification loss for the input batch
    """

    with tf.GradientTape() as tape:

        true_shape_list = []
        preprocessed_images = []

        for img in image_list:

            preprocessed_img, true_shape = model.preprocess(img)
            preprocessed_images.append(preprocessed_img)
            true_shape_list.append(true_shape)

        # Make a prediction
        preprocessed_image_tensor = tf.concat(preprocessed_images, axis=0)
        true_shape_tensor = tf.concat(true_shape_list, axis=0)

        prediction_dict = model.predict(preprocessed_inputs=preprocessed_image_tensor,
                                        true_image_shapes=true_shape_tensor)

        # Prodive groundtruth boxes and classes and calculate the total loss (sum of both losses)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)

        loss_dict = model.loss(
            prediction_dict=prediction_dict, true_image_shapes=true_shape_tensor)
        total_loss = loss_dict['Loss/localization_loss'] + \
            loss_dict['Loss/classification_loss']

        # Calculate the gradients
        gradients = tape.gradient([total_loss], vars_to_fine_tune)

        # Optimize the model's selected variables
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))

    loc_loss = loss_dict['Loss/localization_loss']
    class_loss = loss_dict['Loss/classification_loss']

    return total_loss, loc_loss, class_loss


def detectionTrain(
    model, optimizer, to_fine_tune, train_files_path, batch_size, num_epochs, num_classes, 
    label_id_offset, permutations, bbox_format, meta, checkpoint_save_dir, strategy):

    train_filenames = os.listdir(train_files_path)

    steps_per_epoch_train = int(len(train_filenames) // batch_size)

    for epoch in range(num_epochs):

        train_filenames = shuffle(train_filenames)

        for step in range(steps_per_epoch_train):

            train_filenames_batched = train_filenames[step *
                                                      batch_size:(step + 1) * batch_size]

            train_images_batched, train_boxes_batched, train_classes_batched = prepareDetectionDataset(
                train_filenames_batched, train_files_path, num_classes, label_id_offset, permutations, bbox_format, meta)

            total_loss, loc_loss, class_loss = detectionTrainStep(train_images_batched, train_boxes_batched, train_classes_batched,
                                                                  model, optimizer, to_fine_tune)

        print('STEP ' + str(step) + ' OF ' + str(steps_per_epoch_train) + ', loss=' + str(total_loss.numpy()) +
              ' | loc_loss=' + str(loc_loss.numpy()) + ' | class_loss=' + str(class_loss.numpy()), flush=True)

    checkpoint_save_dir_epoch = checkpoint_save_dir + str(epoch) + '_loss-' + str(loc_loss.numpy())
    if not os.path.exists(checkpoint_save_dir_epoch):
        os.makedirs(checkpoint_save_dir_epoch)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory=checkpoint_save_dir_epoch, max_to_keep=1)
    manager.save()

    print('EPOCH ' + str(epoch) + ' OF ' + str(num_epochs) +
          ', loss=' + str(total_loss.numpy()), flush=True)