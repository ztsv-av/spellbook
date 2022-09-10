import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import torch.nn as nn


def categoricalFocalLossWrapper(reduction, alpha=0.25, gamma=2.0):
    """
    softmax version of focal loss
    if there is skew between different categories/labels in dataset you can try to apply this function as a loss

    references
        https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    example of usage
        model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)

    parameters
    ----------
        alpha : float, default is 0.25
            the same as weighing factor in balanced cross entropy
            alpha is used to specify the weight of different categories/labels
            the size of the array needs to be consistent with the number of classes

        gamma : float, default is 2.0
          focusing parameter for modulating factor / relaxation parameter
          more the value of gamma, more importance will be given to misclassified examples and very less loss will be propagated from easy examples.

    returns
    -------
        categorical_focal_loss_fixed : function
          function that that returns mean of focal loss for given batch
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categoricalFocalLoss(y_true, y_pred):
        """
        calculates mean of focal loss between true and predicted labels

        parameters
        ----------
            y_true : tensor
                tensor of the same shape as `y_pred`

            y_pred : tensor
              tensor resulting from a softmax

        returns
        -------
            mean_loss : output tensor
              mean of focal loss
        """

        # clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # calculate cross-entropy
        cross_entropy = -y_true * K.log(y_pred)

        # calculate focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # compute mean loss in mini batch
        mean_loss = K.mean(K.sum(loss, axis=-1))

        if reduction == None:

            mean_loss = tf.expand_dims(mean_loss, axis=-1)

        return mean_loss

    return categoricalFocalLoss


def binaryFocalLossWrapper(reduction, alpha=0.25, gamma=2.0):
    """
    binary form of focal loss

    references
       https://arxiv.org/pdf/1708.02002.pdf

    example of usage
        model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    parameters
    ----------
        alpha : float, default is 0.25
            the same as weighing factor in balanced cross entropy
            alpha is used to specify the weight of different categories/labels
            the size of the array needs to be consistent with the number of classes

        gamma : float, default is 2.0
            relaxation parameter
            more the value of gamma, more importance will be given to misclassified examples 
            and very less loss will be propagated from easy examples.

      returns
      -------
        binary_focal_loss_fixed : function
            function that returns mean of binary loss for given batch
    """

    def binaryFocalLoss(y_true, y_pred):
        """
        calculates mean of binary focal loss between true and predicted labels

        parameters
        ----------
            y_true : tensor
                true labels

            y_pred : tensor
                predicted labels

        returns
        -------
            mean_loss : output tensor
                mean of focal binary loss
        """

        y_true = tf.cast(y_true, tf.float32)

        # define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()

        # add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # see formula in reference for more
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)

        # calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)

        # calculate focal loss
        loss = weight * cross_entropy

        # sum the losses in mini batch
        mean_loss = K.mean(K.sum(loss, axis=1))

        if reduction == None:

            mean_loss = tf.expand_dims(mean_loss, axis=-1)

        return mean_loss

    return binaryFocalLoss


def rootMeanSquaredErrorLossWrapper(reduction):
    """
    calculates RMSE loss between true and predicted labels

    parameters
    ----------
        y_true : tensor
            true labels

        y_pred : tensor
            predicted labels

    returns
    -------
        rmse_loss : output tensor
            RMSE loss
    """

    def rootMeanSquaredErrorLoss(y_true, y_pred):

        rmse_loss =  tf.keras.metrics.mean_squared_error(y_true, y_pred) ** 0.5

        if reduction == None:

            rmse_loss = tf.expand_dims(rmse_loss, axis=-1)

        return rmse_loss

    return rootMeanSquaredErrorLoss


def logisticLoss(y_true, y_pred):
    """
    computes logistic cost function.

    parameters
    ----------
        y_true : array
            array of true output values.

        y_pred : array
            array of predicted values for each true value.

    returns
    -------
        logistic : float
            value of the logistic cost function
    """

    logistic = -1 * y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    logistic = np.mean(logistic)

    return logistic


def mseLoss(y_true, y_pred):
    """
    mean squared error cost function.

    parameters
    ----------
        y_true : array
            array of true output values.

        y_pred : array
            array of predicted values for each true value.

    returns
    -------
        mse : float
            value of the mse
    """

    mse = np.substract(y_true, y_pred)
    mse = np.power(mse, 2)
    mse = np.mean(mse)

    return mse
