from prepareTrainDataset import prepareClassificationDataset, prepareDetectionDataset
from callbacks import LRLadderDecrease ,saveTrainInfo, saveTrainWeights, saveModel, saveTrainInfoDetection, saveCheckpointDetection
from helpers import getFullPaths, loadNumpy, getFeaturesFromPath, getLabelFromPath

import time
import tensorflow as tf
from sklearn.utils import shuffle


def classificationDistributedTrainStepWrapper():
    """
    wrapper for distributed training iteration on a batch of data

    returns
    -------

        classificationDistributedTrainStep : function
            function that computes reduced loss on a batch of data
    """

    @tf.function
    def classificationDistributedTrainStep(inputs, model, compute_total_loss, optimizer, metric_type, train_metric, strategy):
        """
        computes losses on every GPU and reduces (averages) them

        parameters
        ----------

            inputs : tuple
                contains training data and target features

            model : object
                model that is being trained

            compute_total_loss : function
                returns average loss for each loss calculated on each GPU

            optimizer : object
                function or an algorithm that modifies weights and learning rate of a model
            
            metric_type : string
                either a custom metric or imported
            
            train_metric : object/function
                train metric to calculate

            strategy : tf.distribute object
                TensorFlow API used in distributed training

        returns
        -------

            reduced_loss : tensor
                training loss calculated on a batch of data
                reduced for every used GPU
        """

        per_replica_losses, per_replica_metrics = strategy.run(classificationTrainStep, args=(
            inputs, model, compute_total_loss, optimizer, metric_type, train_metric))

        reduced_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        reduced_loss /= strategy.num_replicas_in_sync

        reduced_metric = 0.0
        if metric_type == 'custom':
            reduced_metric = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_metrics, axis=None)
        reduced_metric /= strategy.num_replicas_in_sync
        

        # test if per replica training works
        # tf.print(per_replica_losses.values)
        # tf.print(reduced_loss)
        # tf.print(per_replica_metrics.values)
        # tf.print(reduced_metric)

        return reduced_loss, reduced_metric

    return classificationDistributedTrainStep


def classificationTrainStep(inputs, model, compute_total_loss, optimizer, metric_type, train_metric):
    """
    computes loss on a batch of data, performs gradient descent to train a model and updates training metrics

    parameters
    ----------

        inputs : tuple
            contains training data and target features

        model : object
            model that is being trained

        compute_total_loss : function
            returns average loss for each loss calculated on each GPU

        optimizer : object
            function or an algorithm that modifies weights and learning rate of a model
       
        metric_type : string
            either a custom metric or imported
        
        train_metric : object/function
            train metric to calculate

    returns
    -------

        loss : tensor
            total loss of a batch of data
    """

    if type(inputs[0]) is tuple:

        input_data_features = inputs[0]
        data = input_data_features[0]
        features = input_data_features[1:]

    else:

        data = inputs[0]
        features = []
    
    if type(inputs[1]) is tuple:

        input_labels = inputs[1]
        labels = [label for label in input_labels]
    
    else:

        labels = inputs[1]

    with tf.GradientTape() as tape:

        if len(features) == 0:

            prediction_data = data
        
        else:

            prediction_data = []
            prediction_data.append(data)

            for feature in features:

                prediction_data.append(feature)

        predictions = model(prediction_data, training=True)

        labels_concat = tf.concat(labels, axis=1)
        predictions_concat = tf.concat(predictions, axis=1)

        loss = compute_total_loss(labels_concat, predictions_concat)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if metric_type == 'custom':

        custom_train_metric = train_metric(labels, predictions)

        return loss, custom_train_metric

    else:

        train_metric.update_state(labels_concat, predictions_concat)

        return loss, 0.0


def classificationDistributedValStepWrapper():
    """
    wrapper for distributed validation iteration on a batch of data

    returns
    -------

        classificationDistributedValStep : function
            function that calls another function to compute validation loss on a batch of data
    """

    @tf.function
    def classificationDistributedValStep(inputs, model, loss_object, val_loss, metric_type, val_metric, strategy):
        """
        calls classificationValStep function to compute validation loss on a batch of data and update validation metrics
        no need to reduce the loss because it is being updated inside classificationValStep function

        parameters
        ----------

            inputs : tuple
                contains training data and target features

            model : object
                model that is being validated

            loss_object : object/function
                computes loss between true and predicted labels

            val_loss : object
                validation loss to update
                usually just a mean
            
            metric_type : string
                either a custom metric or imported
            
            val_metric : object/function
                validation metric to calculate

            strategy : tf.distribute object
                TensorFlow API used in distributed training

        returns
        -------

            strategy.run(classificationValStep) : call to a function
                calls a function that computes and updates states of validation loss and metrics
        """

        per_replica_metrics = strategy.run(classificationValStep, args=(inputs, model, loss_object, val_loss, metric_type, val_metric))

        reduced_metric = 0.0
        if metric_type == 'custom':
            reduced_metric = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_metrics, axis=None)
        reduced_metric /= strategy.num_replicas_in_sync

        return reduced_metric

    return classificationDistributedValStep


def classificationValStep(inputs, model, loss_object, val_loss, metric_type, val_metric):
    """
    computes loss on a batch of data and updates states of validation loss and metrics

    parameters
    ----------

        inputs : tuple
            contains training data and target features

        model : object
            model that is being validated

        loss_object : object/function
            computes loss between true and predicted labels

        val_loss : object
            validation loss to update
            usually just a mean
        
        metric_type : string
            either a custom metric or imported
        
        val_metric : object/function
            validation metric to calculate
    """

    if type(inputs[0]) is tuple:

        input_data_features = inputs[0]
        data = input_data_features[0]
        features = input_data_features[1:]

    else:

        data = inputs[0]
        features = []
    
    if type(inputs[1]) is tuple:

        input_labels = inputs[1]
        labels = [label for label in input_labels]
    
    else:

        labels = inputs[1]

    if len(features) == 0:

        prediction_data = data
    
    else:

        prediction_data = []
        prediction_data.append(data)

        for feature in features:

            prediction_data.append(feature)

    predictions = model(prediction_data, training=False)
    
    labels_concat = tf.concat(labels, axis=1)
    predictions_concat = tf.concat(predictions, axis=1)

    val_batch_loss = loss_object(labels_concat, predictions_concat)

    val_loss.update_state(val_batch_loss)

    if metric_type == 'custom':

        custom_val_metric = val_metric(labels_concat, predictions_concat)

        return custom_val_metric

    else:

        val_metric.update_state(labels_concat, predictions_concat)

        return 0.0


def classificationCustomTrain(
        num_epochs, start_epoch, batch_size, num_classes, num_add_classes,  
        train_paths_list, val_paths_list, do_validation, max_fileparts_train, max_fileparts_val, fold,
        metadata, id_column, feature_columns, add_features_columns, 
        filename_underscore, create_onehot, onehot_idx, onehot_idxs_add,   
        permutations, do_permutations, normalization,
        model_name, model,
        loss_object, val_loss, compute_total_loss,
        lr_ladder, lr_ladder_step, lr_ladder_epochs, optimizer,
        metric_type, train_metric, val_metric,
        save_train_info_dir, save_train_weights_dir,
        strategy):
    """
    main training function for classification/encoding-decoding tasks
    for each epoch it divides training and validation files into parts
    for every part it preprocessed data, computes loss and metrics
    after an epoch is finished, it saves the training information and model weights
    as well as updates loss and metric states of the model and learning rate

    parameters
    ----------

        num_epochs : integer
            number of epochs to train the model

        start_epoch : integer
            number of the epoch from which the training process starts

        batch_size : integer
            number of training examples in one batch of data

        num_classes : integer
            number of classes

        num_add_classes : list
            contains integers representing number of classes in additional features

        train_paths_list : list
            full paths to train files

        val_paths_list : list
            full paths to validation files

        do_validation : boolean
            do validation or not

        max_fileparts_train : integer
            maximum number of training parts
            one part has a certain number of files in it
            used to solve memory allocation error

        max_fileparts_val : integer
            maximum number of validation parts
            one part has a certain number of files in it
            used to solve memory allocation error

        fold : integer
            fold number

        metadata : dataframe
            table containing ids of files, additional features and features to predict

        id_column : string
            name of the id column in the metadata

        feature_columns : list
            names of target feature columns

        add_features_columns : list
            names of additional feature columns to add as an input when training

        filename_underscore : boolean
            True if end of a filename has a class after an underscore
            used to properly extract data from metadata

        create_onehot : boolean
            whether to use metadata and load one-hot vector from there
            or create a new one using a name of the file

        onehot_idx : 
            which idx to use when splitting filename path by underscore to create one-hot vector

        onehot_idxs_add : list
             contains indicies to use when splitting filename path by underscore to create one-hot vectors for additional features

        permutations : list
            list of data permutation functions

        do_permutations : boolean
            either to perfrom data permutations or not

        normalization : function
            normalization function to apply to data

        model_name : string
            name of the model

        model : object
            model to train

        loss_object : object/function
            computes loss between true and predicted labels

        val_loss : object
            validation loss to update
            usually just a mean

        compute_total_loss : function
            returns average loss for each loss calculated on each GPU

        lr_ladder : boolean
            either to perform ladder learning rate reduction or not

        lr_ladder_step : decimal
            ladder learning rate reductuion rate

        lr_ladder_epochs : integer
            number of epochs to pass to perform the ladder learning rate reduction

        optimizer : object
            function or an algorithm that modifies weights and learning rate of a model

        metric_type : string
            either a custom metric or imported

        train_metric : object/function
            train metric to calculate

        val_metric : object/function
            validation metric to calculate

        save_train_info_dir : string
            full path to directory where to save training information, such as epoch number, training loss, training accuracy, etc.

        save_train_weights_dir : string
            full path to directory where to save model weights

        strategy : tf.distribute object
            TensorFlow API used in distributed training
    """

    wrapperTrain = classificationDistributedTrainStepWrapper()
    wrapperVal = classificationDistributedValStepWrapper()

    for epoch in range(start_epoch, num_epochs):

        train_paths_list_shuffled = shuffle(train_paths_list)
        if do_validation:
            val_paths_list_shuffled = shuffle(val_paths_list)

        total_loss = 0.0
        total_train_metric = 0.0
        num_train_batches = 0

        for part in range(max_fileparts_train):

            start_part_time = time.time()

            train_filepaths_part = train_paths_list_shuffled[
                int(part * len(train_paths_list_shuffled) / max_fileparts_train) :
                int(((part + 1) / max_fileparts_train) * len(train_paths_list_shuffled))]

            start_load_data_time = time.time()
            if fold == None:
                print('\nEpoch ' + str(epoch + 1) + '. Loading training data...', flush=True)
            else:
                print('\nEpoch ' + str(epoch + 1) + '. Fold ' + str(fold + 1) + '. Loading training data...', flush=True)

            train_distributed_part = prepareClassificationDataset(
                batch_size, num_classes, num_add_classes,  
                train_filepaths_part, 
                metadata, id_column, feature_columns, add_features_columns, 
                filename_underscore, create_onehot, onehot_idx, onehot_idxs_add,   
                permutations, do_permutations, normalization, 
                strategy, is_val=False)

            end_load_data_time = time.time()
            if fold == None:
                print('\nEpoch ' + str(epoch + 1) + '. Finished loading training data. Time passed: ' 
                    + str(end_load_data_time - start_load_data_time), flush=True)
            else:
                print('\nEpoch ' + str(epoch + 1) + '. Fold ' + str(fold + 1) + '. Finished loading training data. Time passed: ' 
                    + str(end_load_data_time - start_load_data_time), flush=True)

            for batch in train_distributed_part:

                batch_loss, batch_train_metric = wrapperTrain(
                    batch, model, compute_total_loss, optimizer, metric_type, train_metric, strategy)

                total_loss += batch_loss
                total_train_metric += batch_train_metric

                num_train_batches += 1

            end_part_time = time.time()
            if fold == None:
                print('\nEpoch ' + str(epoch + 1) + '. Training: part ' + str(part + 1) + '/' + str(max_fileparts_train) +
                    ', passed time: ' + str(end_part_time - start_part_time), flush=True)
            else:
                print('\nEpoch ' + str(epoch + 1) + '. Fold ' + str(fold + 1) + '. Training: part ' + str(part + 1) + '/' + str(max_fileparts_train) +
                    ', passed time: ' + str(end_part_time - start_part_time), flush=True)

        train_loss = total_loss / num_train_batches
        custom_train_metric = total_train_metric / num_train_batches

        del train_distributed_part

        total_val_metric = 0.0
        num_val_batches = 0

        if do_validation:

            for part in range(max_fileparts_val):

                start_part_time = time.time()

                val_filepaths_part = val_paths_list_shuffled[
                    int(part * len(val_paths_list_shuffled) / max_fileparts_val) :
                    int(((part + 1) / max_fileparts_val) * len(val_paths_list_shuffled))]

                start_load_data_time = time.time()
                if fold == None:
                    print('\nEpoch ' + str(epoch + 1) + '. Loading validation data...', flush=True)
                else:
                    print('\nEpoch ' + str(epoch + 1) + '. Fold ' + str(fold + 1) + '. Loading validation data...', flush=True)

                val_distributed_part = prepareClassificationDataset(
                    batch_size, num_classes, num_add_classes, 
                    val_filepaths_part, 
                    metadata, id_column, feature_columns, add_features_columns, 
                    filename_underscore, create_onehot, onehot_idx, onehot_idxs_add, 
                    None, do_permutations, normalization, 
                    strategy, is_val=True)

                end_load_data_time = time.time()
                if fold == None:
                    print('\nEpoch ' + str(epoch + 1) + '. Finished validation loading data. Time passed: ' 
                        + str(end_load_data_time - start_load_data_time), flush=True)
                else:
                    print('\nEpoch ' + str(epoch + 1) + '. Fold ' + str(fold + 1) + '. Finished validation loading data. Time passed: ' 
                        + str(end_load_data_time - start_load_data_time), flush=True)

                for batch in val_distributed_part:

                    batch_val_metric = wrapperVal(
                        batch, model, loss_object, val_loss, metric_type, val_metric, strategy)

                    total_val_metric += batch_val_metric

                    num_val_batches += 1

                end_part_time = time.time()
                if fold == None:
                    print('\nEpoch ' + str(epoch + 1) + '. Validation: part ' + str(part + 1) + '/' + str(max_fileparts_val) +
                        ', passed time: ' + str(end_part_time - start_part_time), flush=True)
                else:
                    print('\nEpoch ' + str(epoch + 1) + '. Fold ' + str(fold + 1) + '. Validation: part ' + str(part + 1) + '/' + str(max_fileparts_val) +
                        ', passed time: ' + str(end_part_time - start_part_time), flush=True)

            custom_val_metric = total_val_metric / num_val_batches

            del val_distributed_part

            template = (
                "Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}")
            
            if metric_type == 'custom':
            
                print('\n' + template.format(
                    epoch + 1, train_loss, custom_train_metric * 100,
                    val_loss.result(), custom_val_metric * 100, flush=True))
                
                saveTrainInfo(
                    model_name, epoch, fold, 
                    train_loss, val_loss, 
                    metric_type, custom_train_metric, custom_val_metric, 
                    optimizer, save_train_info_dir)

            else:

                print('\n' + template.format(
                    epoch + 1, train_loss, train_metric.result() * 100,
                    val_loss.result(), val_metric.result() * 100, flush=True))

                saveTrainInfo(
                model_name, epoch, fold, 
                train_loss, val_loss, 
                metric_type, train_metric, val_metric, 
                optimizer, save_train_info_dir)

            saveModel(model, model_name, epoch, fold, save_train_weights_dir)

            if lr_ladder:

                if ((epoch + 1) % lr_ladder_epochs == 0):

                    new_lr = LRLadderDecrease(optimizer, lr_ladder_step)

                    optimizer.learning_rate = new_lr

            val_loss.reset_states()
            if metric_type == 'custom':
                custom_train_metric = 0.0
                custom_val_metric = 0.0
            else:
                train_metric.reset_states()
                val_metric.reset_states()

            # sleep 120 seconds
            if ((epoch + 1) % 10 == 0):

                print('\nSleeping 120 seconds after every 10 epochs. Zzz...', flush=True)
                time.sleep(120)
        
        else:

            template = (
                "Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}")
            
            if metric_type == 'custom':
            
                print('\n' + template.format(
                    epoch + 1, train_loss, train_metric * 100,
                    'No validation', 'No validation', flush=True))

                saveTrainInfo(
                model_name, epoch, fold, 
                train_loss, None, 
                metric_type, custom_train_metric, None, 
                optimizer, save_train_info_dir)
                
            else:

                print('\n' + template.format(
                    epoch + 1, train_loss, train_metric.result() * 100,
                    'No validation', 'No validation', flush=True))

                saveTrainInfo(
                model_name, epoch, fold, 
                train_loss, None, 
                metric_type, train_metric, None, 
                optimizer, save_train_info_dir)

            saveModel(model, model_name, epoch, fold, save_train_weights_dir)

            if lr_ladder:

                if ((epoch + 1) % lr_ladder_epochs == 0):

                    new_lr = LRLadderDecrease(optimizer, lr_ladder_step)

                    optimizer.learning_rate = new_lr

            if metric_type == 'custom':
                custom_train_metric = 0.0
            else:
                train_metric.reset_states()

            # sleep 120 seconds
            if ((epoch + 1) % 10 == 0):

                print('\nSleeping 120 seconds after every 10 epochs. Zzz...', flush=True)
                time.sleep(120)


def detectionTrainStep(
        image_list, groundtruth_boxes_list, groundtruth_classes_list,
        model, vars_to_fine_tune, optimizer):
    """
    single object detection training iteration
    computes localization and classification losses for input batches of data
    and performs gradient descent to train the model parameters

    parameters
    ----------
        image_list: array
            array of [height, width, 3] tensors of type tf.float32
            images reshaped to model's preprocess function

        groundtruth_boxes_list: array
            array of tensors of shape [num_boxes, 4] with type tf.float32 representing groundtruth boxes for each image

        groundtruth_classes_list: array
            list of tensors of shape [num_boxes, num_classes] with type tf.float32 representing groundtruth boxes for each image
        
        model : object
            model to train
        
        vars_to_fine_tune : list
            list of trainable variables ('WeightSharedConvolutionalBoxPredictor' variables) of the model
        
        optimizer : object
            function or an algorithm that modifies weights and learning rate of a model
            

    returns
    -------
        total_loss: scalar tensor
            represents total loss for input data

        loc_loss: scalar tensor
            represents localization loss for input data

        class_loss: scalar tensor
            represents classification loss for input data
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

        # provide groundtruth boxes and classes and calculate the total loss (sum of both losses)
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
        num_epochs, batch_size,
        num_classes, label_id_offset,
        train_filepaths, bbox_format, meta, 
        permutations, normalization,
        model, model_name,
        optimizer,to_fine_tune, 
        checkpoint_save_dir, save_train_info_dir):
    """
    main training function for object detection and classification tasks
    for each epoch it split data in parts
    for each part it loads and preprocesses the data,
    computes localization and classification losses
    and updates their states
    it minimizes loss functions by finding their local minimum/maximum,
    i.e. by performing a gradient descent

    parameters
    ----------

    num_epochs : integer
        number of epochs to train the model

    batch_size : integer
        number of training examples in one batch of data

    num_classes : integer
        number of classes to predict

    label_id_offset : integer
        shifts all classes by a certain number of indices
        so that the model receives one-hot labels where non-background
        classes start counting at the zeroth index

    train_filepaths : list
        full paths to files

    bbox_format : string
        format of bounding boxes

    meta : dataframe
        metadata, table containing ids of files, additional features, features to predict and coordintates of bounding boxes

    permutations : list
        list of data permutation functions
        
    normalization : function
        normalization function to apply to data

    model : object
        model to train

    model_name : string
        name of the model

    optimizer : object
        function or an algorithm that modifies weights and learning rate of a model

    to_fine_tune : list
        list of trainable variables ('WeightSharedConvolutionalBoxPredictor' variables) of the model

    checkpoint_save_dir : string
        full path to directory where to save the training checkpoint (model weights)

    save_train_info_dir : string
        full path to directory where to save training information, that is epoch number, loss values and learning rate
    """

    train_filepaths_list = getFullPaths(train_filepaths)

    steps_per_epoch_train = int(len(train_filepaths_list) // batch_size)

    for epoch in range(num_epochs):

        train_filepaths_list_shuffled = shuffle(train_filepaths_list)

        for step in range(steps_per_epoch_train):

            train_filepaths_batched = train_filepaths_list_shuffled[
                step * batch_size:(step + 1) * batch_size]

            train_images_batched, train_boxes_batched, train_classes_batched = prepareDetectionDataset(
                train_filepaths_batched, bbox_format, meta, num_classes, label_id_offset, permutations,
                normalization, is_val=False)

            total_loss, loc_loss, class_loss = detectionTrainStep(
                train_images_batched, train_boxes_batched, train_classes_batched,
                model, to_fine_tune, optimizer)

            print('STEP ' + str(step) + ' OF ' + str(steps_per_epoch_train) + ', loss=' + str(total_loss.numpy()) +
                  ' | loc_loss=' + str(loc_loss.numpy()) + ' | class_loss=' + str(class_loss.numpy()), flush=True)

        saveTrainInfoDetection(model_name, epoch, loc_loss,
                               class_loss, total_loss, optimizer, save_train_info_dir)
        saveCheckpointDetection(model_name, epoch, model,
                                loc_loss, optimizer, checkpoint_save_dir)

        print('EPOCH ' + str(epoch) + ' OF ' + str(num_epochs) +
              ', loss=' + str(total_loss.numpy()), flush=True)
