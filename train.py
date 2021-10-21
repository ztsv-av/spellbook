from prepareTrainDataset import prepareClassificationDataset, prepareDetectionDataset
from callbacks import saveTrainInfo, saveTrainWeights, saveTrainInfoDetection, saveCheckpointDetection

import os
import tensorflow as tf
from sklearn.utils import shuffle

# CLASSIFICATION


def classificationTrainStep(inputs, model, compute_total_loss, optimizer, train_accuracy):
    """
    computes total loss after one train step

    parameters
    ----------

        inputs : XXX
            XXX

        model : XXX
            XXX

        compute_total_loss : XXX
            XXX

        optimizer : XXX
            XXX

        train_accuracy : XXX
            XXX

    returns
    -------

        loss : XXX
            total loss after train step
    """

    images, labels = inputs

    with tf.GradientTape() as tape:

        predictions = model(images, training=True)
        loss = compute_total_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)

    return loss


def classificationValStep(inputs, model, loss_object, val_loss, val_accuracy):
    """
    updates loss and accuracy after validation step for classification

    parameters
    ----------
        inputs : XXX
            XXX

        model : XXX
            XXX

        loss_object : XXX
            XXX

        val_loss : XXX
            XXX

        val_accuracy : XXX
            XXX
    """

    images, labels = inputs

    predictions = model(images, training=False)
    val_batch_loss = loss_object(labels, predictions)

    val_loss.update_state(val_batch_loss)
    val_accuracy.update_state(labels, predictions)


def classificationDistributedTrainStepWrapper():
    """
    wrapper for distributed train step for classification

    returns
    -------

        classificationDistributedTrainStep : function
            function that computes reduced loss after distributed train step
    """

    @tf.function
    def classificationDistributedTrainStep(inputs, model, compute_total_loss, optimizer, train_accuracy, strategy):
        """
        computes reduced loss after one distributed train step

        parameters
        ----------

            inputs : XXX
                XXX

            model : XXX
                XXX

            compute_total_loss : XXX
                XXX

            optimizer : XXX
                XXX

            train_accuracy : XXX
                XXX

            strategy : XXX
                XXX

        returns
        -------

            reduced_loss : XXX
                XXX
        """

        per_replica_losses = strategy.run(classificationTrainStep, args=(
            inputs, model, compute_total_loss, optimizer, train_accuracy))

        reduced_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        # test if per replica training works
        # tf.print(per_replica_losses.values)
        # tf.print(reduced_loss)

        return reduced_loss

    return classificationDistributedTrainStep


def classificationDistributedValStepWrapper():
    """
    XXX

    returns
    -------

        classificationDistributedValStep : XXX
            XXX
    """

    @tf.function
    def classificationDistributedValStep(inputs, model, loss_object, val_loss, val_accuracy, strategy):
        """
        XXX

        parameters
        ----------

            inputs : XXX
                XXX

            model : XXX
                XXX

            loss_object : XXX
                XXX

            val_loss : XXX
                XXX

            val_accuracy : XXX
                XXX

            strategy : XXX
                XXX

        returns
        -------

            XXX : XXX
                XXX
        """

        return strategy.run(classificationValStep, args=(inputs, model, loss_object, val_loss, val_accuracy))

    return classificationDistributedValStep


def classificationCustomTrain(
        batch_size, num_epochs, train_data, val_data, permutations, normalization, buffer_size, model, loss_object,
        val_loss, compute_total_loss, optimizer, train_accuracy, val_accuracy, save_csvs_dir, save_weights_dir,
        model_name, strategy):
    """
    XXX

    parameters
    ----------

        batch_size : XXX
            XXX

        num_epochs : XXX
            XXX

        train_data : XXX
            XXX

        val_data : XXX
            XXX

        permutations : XXX
            XXX

        normalization : XXX
            XXX

        buffer_size : XXX
            XXX

        model : XXX
            XXX

        loss_object : XXX
            XXX

        val_loss : XXX
            XXX

        compute_total_loss : XXX
            XXX

        optimizer : XXX
            XXX

        train_accuracy : XXX
            XXX

        val_accuracy : XXX
            XXX

        save_csvs_dir : XXX
            XXX

        save_weights_dir : XXX
            XXX

        model_name : XXX
            XXX

        strategy : XXX
            XXX
    """
    wrapperTrain = classificationDistributedTrainStepWrapper()
    wrapperVal = classificationDistributedValStepWrapper()

    for epoch in range(num_epochs):

        train_distributed_dataset, val_distributed_dataset = prepareClassificationDataset(
            batch_size, train_data, val_data, permutations, normalization, buffer_size, strategy)

        total_loss = 0.0
        num_batches = 0

        for batch in train_distributed_dataset:

            total_loss += wrapperTrain(
                batch, model, compute_total_loss, optimizer, train_accuracy, strategy)
            num_batches += 1

        train_loss = total_loss / num_batches

        for batch in val_distributed_dataset:

            wrapperVal(
                batch, model, loss_object, val_loss, val_accuracy, strategy)

        template = (
            "Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, " "Validation Accuracy: {}")
        print(template.format(
            epoch + 1, train_loss, train_accuracy.result() * 100,
            val_loss.result(), val_accuracy.result() * 100, flush=True))

        # callbacks
        saveTrainInfo(model_name, epoch, train_loss, train_accuracy,
                      val_loss, val_accuracy, optimizer, save_csvs_dir)
        saveTrainWeights(model, model_name, epoch, save_weights_dir)

        val_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()

# DETECTION


def detectionTrainStep(
        image_list, groundtruth_boxes_list, groundtruth_classes_list,
        model, vars_to_fine_tune, optimizer):
    """
    single training iteration

    parameters
    ----------
        image_list: array
            array of [1, height, width, 3] Tensor of type tf.float32
            images reshaped to model's preprocess function

        groundtruth_boxes_list: array
            array of Tensors of shape [num_boxes, 4] with type tf.float32 representing groundtruth boxes for each image in batch

        groundtruth_classes_list: array
            list of Tensors of shape [num_boxes, num_classes] with type tf.float32 representing groundtruth boxes for each image in batch

    returns
    -------
        total_loss: scalar tensor
            represents total loss for input batch

        loc_loss: scalar tensor
            represents localization loss for input batch

        class_loss: scalar tensor
            represents classification loss for input batch
    """

    with tf.GradientTape() as tape:

        true_shape_list = []
        preprocessed_images = []

        for img in image_list:

            preprocessed_img, true_shape = model.preprocess(img)
            preprocessed_images.append(preprocessed_img)
            true_shape_list.append(true_shape)

        # make prediction
        preprocessed_image_tensor = tf.concat(preprocessed_images, axis=0)
        true_shape_tensor = tf.concat(true_shape_list, axis=0)

        prediction_dict = model.predict(
            preprocessed_inputs=preprocessed_image_tensor,
            true_image_shapes=true_shape_tensor)

        # prodive groundtruth boxes and classes and calculate the total loss (sum of both losses)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)

        loss_dict = model.loss(
            prediction_dict=prediction_dict, true_image_shapes=true_shape_tensor)
        total_loss = loss_dict['Loss/localization_loss'] + \
            loss_dict['Loss/classification_loss']

        # calculate gradients
        gradients = tape.gradient([total_loss], vars_to_fine_tune)

        # optimize model's selected variables
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))

    loc_loss = loss_dict['Loss/localization_loss']
    class_loss = loss_dict['Loss/classification_loss']

    return total_loss, loc_loss, class_loss


def detectionTrain(
        batch_size, num_epochs, num_classes, label_id_offset,
        train_filepaths, bbox_format, meta, permutations,
        normalization, model, model_name, optimizer,
        to_fine_tune, checkpoint_save_dir, save_csvs_dir):
    """
    XXX

    parameters
    ----------

    batch_size : XXX
        XXX

    num_epochs : XXX
        XXX

    num_classes : XXX
        XXX

    label_id_offset : XXX
        XXX

    train_filepaths : XXX
        XXX

    bbox_format : XXX
        XXX

    meta : XXX
        XXX

    permutations : XXX
        XXX

    normalization : XXX
        XXX

    model : XXX
        XXX

    model_name : XXX
        XXX

    optimizer : XXX
        XXX

    to_fine_tune : XXX
        XXX

    checkpoint_save_dir : XXX
        XXX

    save_csvs_dir : XXX
        XXX
    """

    train_filepaths_list = [train_filepaths +
                            filename for filename in os.listdir(train_filepaths)]

    steps_per_epoch_train = int(len(train_filepaths_list) // batch_size)

    for epoch in range(num_epochs):

        train_filepaths_list_shuffled = shuffle(train_filepaths_list)

        for step in range(steps_per_epoch_train):

            train_filepaths_batched = train_filepaths_list_shuffled[
                step * batch_size:(step + 1) * batch_size]

            train_images_batched, train_boxes_batched, train_classes_batched = prepareDetectionDataset(
                train_filepaths_batched, bbox_format, meta, num_classes, label_id_offset, permutations,
                normalization)

            total_loss, loc_loss, class_loss = detectionTrainStep(
                train_images_batched, train_boxes_batched, train_classes_batched,
                model, to_fine_tune, optimizer)

            print('STEP ' + str(step) + ' OF ' + str(steps_per_epoch_train) + ', loss=' + str(total_loss.numpy()) +
                  ' | loc_loss=' + str(loc_loss.numpy()) + ' | class_loss=' + str(class_loss.numpy()), flush=True)

        saveTrainInfoDetection(model_name, epoch, loc_loss,
                               class_loss, total_loss, optimizer, save_csvs_dir)
        saveCheckpointDetection(model_name, epoch, model,
                                loc_loss, optimizer, checkpoint_save_dir)

        print('EPOCH ' + str(epoch) + ' OF ' + str(num_epochs) +
              ', loss=' + str(total_loss.numpy()), flush=True)
