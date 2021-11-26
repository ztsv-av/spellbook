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


def LRLadderDecrease(optimizer, step):

    current_lr = optimizer.learning_rate

    new_lr = current_lr * step

    return new_lr



def saveTrainInfo(model_name, epoch, fold_num, train_loss, train_accuracy, val_loss, val_accuracy, optimizer, save_train_info_dir):

    if val_loss is not None:

        info_df = pd.DataFrame({
            'epoch': [epoch + 1],
            'learning_rate': [optimizer._decayed_lr(tf.float32).numpy()],
            'train_loss': [train_loss.numpy()],
            'train_accuracy': [(train_accuracy.result() * 100).numpy()],
            'val_loss': [val_loss.result().numpy()],
            'val_accuracy': [(val_accuracy.result() * 100).numpy()]})
    
    else:

        info_df = pd.DataFrame({
            'epoch': [epoch + 1],
            'learning_rate': [optimizer._decayed_lr(tf.float32).numpy()],
            'train_loss': [train_loss.numpy()],
            'train_accuracy': [(train_accuracy.result() * 100).numpy()],
            'val_loss': ['No validation'],
            'val_accuracy': ['No validation']})

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

    if fold_num is not None:
        save_weights_epoch_dir = save_train_weights_dir + \
            model_name + '/' + 'fold-' + str(fold_num + 1) + '/' + str(epoch + 1) + '/'
    else:
        save_weights_epoch_dir = save_train_weights_dir + \
            model_name + '/' + 'no-folds' + '/' + str(epoch + 1) + '/'

    if not os.path.exists(save_weights_epoch_dir):
        os.makedirs(save_weights_epoch_dir)

    model.save_weights(save_weights_epoch_dir + 'weights.h5')

def saveModel(model, model_name, epoch, fold_num, save_train_weights_dir):

    if fold_num is not None:
        save_weights_epoch_dir = save_train_weights_dir + \
            model_name + '/' + 'fold-' + str(fold_num + 1) + '/' + str(epoch + 1) + '/'
    else:
        save_weights_epoch_dir = save_train_weights_dir + \
            model_name + '/' + 'no-folds/' + '/' + str(epoch + 1) + '/'

    if not os.path.exists(save_weights_epoch_dir):
        os.makedirs(save_weights_epoch_dir)

    model.save(save_weights_epoch_dir + 'savedModel')


def saveTrainInfoDetection(model_name, epoch, loc_loss, class_loss, total_loss, optimizer, save_csv_dir):

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

    checkpoint_save_dir_epoch = checkpoint_save_dir + model_name + \
        '/' + str(epoch) + '_loss-' + str(loc_loss.numpy())

    if not os.path.exists(checkpoint_save_dir_epoch):
        os.makedirs(checkpoint_save_dir_epoch)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

    manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory=checkpoint_save_dir_epoch, max_to_keep=1)

    manager.save()
