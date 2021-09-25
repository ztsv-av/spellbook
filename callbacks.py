import tensorflow as tf
import datetime


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
