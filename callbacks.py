import tensorflow as tf

import datetime
import pandas as pd
import os


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


def saveTrainInfo(model_name, epoch, train_loss, train_accuracy, val_loss, val_accuracy, save_csvs_dir):

    info_df = pd.DataFrame({
        'epoch': [epoch + 1], 'train_loss': [train_loss.numpy()], 'train_accuracy': [(train_accuracy.result() * 100).numpy()],
        'val_loss': [val_loss.result().numpy()], 'val_accuracy': [(val_accuracy.result() * 100).numpy()]})

    save_csv_epoch_dir = save_csvs_dir + model_name + '/'
    info_df.to_csv(path_or_buf=save_csv_epoch_dir + 'info' + str(epoch + 1) + '.csv', index=False)


def saveTrainWeights(model, model_name, epoch, save_weights_dir):

    save_weights_epoch_dir = save_weights_dir + model_name + '/' + str(epoch + 1) + '/'
    if not os.path.exists(save_weights_epoch_dir):
        os.makedirs(save_weights_epoch_dir)
    model.save_weights(save_weights_epoch_dir)