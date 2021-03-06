import os
import datetime
import pandas as pd
import tensorflow as tf


class TimeCallback(tf.keras.callbacks.Callback):
    """
    callback registering timestamps before model begins training on batch and after model finishes training on batch
    """

    def on_train_batch_begin(self, batch, logs=None):
        """
        """
        print(' Training: batch {} begins at {}'.format
              (batch, datetime.datetime.now().strftime("%H:%M:%S")), '\n')

    def on_train_batch_end(self, batch, logs=None):
        """
        """
        print(' Training: batch {} ends at {}'.format
              (batch, datetime.datetime.now().strftime("%H:%M:%S")), '\n')


class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    """
    detects whether the model is overfitting or not
    checks if validation loss divided by the training loss is more than the specified threshold
    if it is, the training stops
    """

    def __init__(self, threshold):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        """
        """
        ratio = logs['val_loss'] / logs['loss']
        print('Epoch: {}, Val/Train Loss Ratio: {:.2f}'.format(epoch, ratio))

        if ratio > self.threshold:
            print('Stopping Training...')
            self.model.stop_training = True


def reduceLRCustom(optimizer, factor):
    """
    decreases optimizer's learning rate in a ladder fashion
    multiplies current learning rate by the specified fraction (factor)

    parameters
    ----------
        optimizer : object
            model's optimizer

        factor : number (usually a rational)
            learning rate multiplier

    returns
    -------
        new_lr : number
            new optimizer's learning rate
    """

    current_lr = optimizer.learning_rate
    new_lr = current_lr * factor

    return new_lr


def reduceLROnPlateau(optimizer, metrics_dict, patience, factor, min_lr, monitor):
    """
    reduces optimizer's learning rate if metric is in stagnation, i.e. reached plateau
    multiplies current learning rate by the specified fraction (factor)
reduce_lr_patience, reduce_lr_factor, reduce_lr_minimal_lr, reduce_lr_metric
    parameters
    ----------
        optimizer : object
            model's optimizer

        metrics_dict : dictionary
            dictionary containing metrics such as
            training loss, validation loss, training accuracy, validation accuracy
            and current patience and minimal metric values
        
        patience : integer
            number of epochs with no improvement after which learning rate will be reduced
            
        factor : rational number
            learning rate multiplier
        
        min_lr : rational number
            minimal value of a learning rate
            if current learning rate is less than or equal to this value,
            function returns current learning rate
        
        monitor : string
            which metric to check if it reached plateau

    returns
    -------
        current_lr : number
            current optimizer's learning rate
        
        or

        new_lr : number
            new optimizer's learning rate
    """

    current_lr = optimizer.learning_rate * 1
    new_lr = current_lr * factor

    if current_lr <= min_lr:
        return current_lr, metrics_dict
    
    metric_to_monitor = metrics_dict[monitor]

    if len(metric_to_monitor) < 1:
        return current_lr, metrics_dict

    last_metric_value = metric_to_monitor[-1]

    if last_metric_value < metrics_dict['min_metric_value']:
        metrics_dict['min_metric_value'] = last_metric_value
    
    else:
        metrics_dict['patience'] += 1
    
    if metrics_dict['patience'] >= patience:
        metrics_dict['patience'] = 0

        return new_lr, metrics_dict
    
    return current_lr, metrics_dict


def LRLadderDecrease(optimizer, factor):
    """
    decreases optimizer's learning rate in a ladder fashion
    multiplies current learning rate by the specified fraction (factor)

    parameters
    ----------
        optimizer : object
            model's optimizer

        factor : number (usually a rational)
            learning rate multiplier

    returns
    -------
        new_lr : number
            new optimizer's learning rate
    """

    current_lr = optimizer.learning_rate
    new_lr = current_lr * factor

    return new_lr


def saveTrainInfo(
    model_name, epoch, fold_num, 
    loss_object, train_loss, val_loss, 
    metric_type, train_metrics, val_metrics, 
    optimizer, save_train_info_dir):
    """
    saves current training information in a dataframe

    parameters
    ----------
        model_name : string
            model name

        epoch : integer
            training epoch number

        fold_num : integer
            training fold number

        loss_object : object
            loss function of the model

        train_loss : number
            training loss of the model

        val_loss : number
            validation loss of the model

        metric_type : string
            either a custom metric or imported

        train_metrics : list
            list of training metrics of the model

        val_metrics : list
            list of validation metrics of the model

        optimizer : object
            model's optimizer

        save_train_info_dir : string
            directory where to save the information dataframe
    """

    if val_loss is not None:

        if metric_type == 'custom':

            info_df = pd.DataFrame({
                'epoch': [epoch + 1],
                'learning_rate': [optimizer._decayed_lr(tf.float32).numpy()],
                'train_loss': [train_loss.numpy()],
                'train_metric': [(train_metrics).numpy()],
                'val_loss': [val_loss.result().numpy()],
                'val_metric': [(val_metrics).numpy()]})

        else:

            info_df = pd.DataFrame({
                'epoch': [epoch + 1],
                'learning_rate': [optimizer._decayed_lr(tf.float32).numpy()],
                'train_loss': [train_loss.numpy()],
                val_loss.name: [val_loss.result().numpy()]})
            
            for train_metric in train_metrics:
                info_df[train_metric.name] = [(train_metric.result()).numpy()]
            for val_metric in val_metrics:
                info_df[val_metric.name] = [(val_metric.result()).numpy()]            
        
    else:

        if metric_type == 'custom':

            info_df = pd.DataFrame({
                'epoch': [epoch + 1],
                'learning_rate': [optimizer._decayed_lr(tf.float32).numpy()],
                'train_loss': [train_loss.numpy()],
                'train_metric': [(train_metrics).numpy()],
                'val_loss': ['No validation'],
                'val_metric': ['No validation']})

        else:

            info_df = pd.DataFrame({
                'epoch': [epoch + 1],
                'learning_rate': [optimizer._decayed_lr(tf.float32).numpy()],
                'train_loss': [train_loss.numpy()],
                'val_loss': ['No validation']})
            
            for train_metric in train_metrics:
                info_df[train_metric.name] = [(train_metric.result()).numpy()]
            info_df['val_metric'] = ['No validation']

    save_csv_dir = save_train_info_dir + model_name + '/'

    if fold_num is not None:
        save_csv_path = save_csv_dir + 'fold-' + str(fold_num + 1) + '_training-info.csv'
    else:
        save_csv_path = save_csv_dir + 'training-info.csv'


    if not os.path.exists(save_csv_dir):
        os.makedirs(save_csv_dir)

    files_in_dir = os.listdir(save_csv_dir)

    if save_csv_path.split('/')[-1] not in files_in_dir:
        info_df.to_csv(path_or_buf=save_csv_path, index=False)

    else:
        old_info_df = pd.read_csv(save_csv_path)
        new_info_df = old_info_df.append(info_df, ignore_index=True)
        new_info_df.to_csv(path_or_buf=save_csv_path, index=False)


def saveTrainWeights(model, model_name, epoch, fold_num, save_train_weights_dir):
    """
    saves model's weights

    parameters
    ----------
        model : object
            nodel which is being trained

        model_name : string
            model name

        epoch : integer
            training epoch number

        fold_num : integer
            training fold number

        save_train_weights_dir : string
            directory where to save the weights
    """

    if fold_num is not None:
        save_weights_epoch_dir = save_train_weights_dir + \
            model_name + '/' + 'fold-' + str(fold_num + 1) + '/' + str(epoch + 1) + '/'
    else:
        save_weights_epoch_dir = save_train_weights_dir + \
            model_name + '/' + 'no-folds' + '/' + str(epoch + 1) + '/'

    if not os.path.exists(save_weights_epoch_dir):
        os.makedirs(save_weights_epoch_dir)

    model.save_weights(save_weights_epoch_dir + 'weights.h5')


def saveModel(model, model_name, epoch, fold_num, save_model_dir):
    """
    saves model's weights, optimizer state and metrics

    parameters
    ----------
        model : object
            nodel which is being trained

        model_name : string
            model name

        epoch : integer
            training epoch number

        fold_num : integer
            training fold number

        save_model_dir : string
            directory where to save the model
    """

    if fold_num is not None:
        save_model_epoch_dir = save_model_dir + \
            model_name + '/' + 'fold-' + str(fold_num + 1) + '/' + str(epoch + 1) + '/'
    else:
        save_model_epoch_dir = save_model_dir + \
            model_name + '/' + 'no-folds/' + '/' + str(epoch + 1) + '/'

    if not os.path.exists(save_model_epoch_dir):
        os.makedirs(save_model_epoch_dir)
    
    # if 'EfficientNet' in model_name:

    #     tf.keras.models.save_model(model, save_model_epoch_dir + 'savedModel', options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
    
    # else:

    model.save(save_model_epoch_dir + 'savedModel')


def saveTrainInfoDetection(model_name, epoch, loc_loss, class_loss, total_loss, optimizer, save_csv_dir):
    """
    saves current training information of an object detection model in a dataframe

    parameters
    ----------
        model_name : string
            model name

        epoch : integer
            training epoch number

        loc_loss : number
            currect localization loss

        class_loss : number
            currect classification loss

        total_loss : number
            localization loss + classification loss

        optimizer : object
            model's optimizer

        save_csv_dir : string
            directory where to save the information dataframe
    """

    info_df = pd.DataFrame({
        'epoch': [epoch + 1],
        'learning_rate': [optimizer._decayed_lr(tf.float32).numpy()],
        'loc_loss': [loc_loss.numpy()],
        'class_loss': [class_loss.numpy()],
        'total_loss': [total_loss.numpy()]})

    save_csv_dir_model = save_csv_dir + model_name + '/'
    save_csv_path = save_csv_dir_model + 'training_info.csv'

    if not os.path.exists(save_csv_dir_model):
        os.makedirs(save_csv_dir_model)

    if len(os.listdir(save_csv_dir_model)) == 0:
        info_df.to_csv(path_or_buf=save_csv_path, index=False)

    else:
        old_info_df = pd.read_csv(save_csv_path)
        new_info_df = old_info_df.append(info_df, ignore_index=True)
        new_info_df.to_csv(path_or_buf=save_csv_path, index=False)


def saveCheckpointDetection(model_name, epoch, model, loc_loss, optimizer, checkpoint_save_dir):
    """
    saves current training information of an object detection model in a dataframe

    parameters
    ----------
        model_name : string
            model name

        epoch : integer
            training epoch number

        model : object
            model that is being trained

        loc_loss : number
            currect localization loss

        optimizer : object
            model's optimizer

        checkpoint_save_dir : string
            directory where to save the training checkpoint
    """

    checkpoint_save_dir_epoch = checkpoint_save_dir + model_name + \
        '/' + str(epoch) + '_loss-' + str(loc_loss.numpy())

    if not os.path.exists(checkpoint_save_dir_epoch):
        os.makedirs(checkpoint_save_dir_epoch)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

    manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory=checkpoint_save_dir_epoch, max_to_keep=1)

    manager.save()
