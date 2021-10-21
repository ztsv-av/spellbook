import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def categorical_focal_loss(alpha=0.25, gamma=2.0):
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
          focusing parameter for modulating factor

    returns
    -------
        categorical_focal_loss_fixed : function
          function that that returns mean of focal loss for given batch
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        calculates mean of focal loss between true and predicted labels

        parameters
        ----------
            y_true : XXX
                tensor of the same shape as `y_pred`

            y_pred : XXX
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

        return mean_loss

    return categorical_focal_loss_fixed


def binary_focal_loss(alpha=0.25, gamma=2.0):
    """
    binary form of focal loss

    references
       https://arxiv.org/pdf/1708.02002.pdf

    example of usage
        model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    parameters
    ----------
        alpha : float, default is 0.25
            XXX

        gamma : float, default is 2.0
            XXX

      returns
      -------
        binary_focal_loss_fixed : function
            function that returns mean of binary loss for given batch
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        calculates mean of binary focal loss between true and predicted labels

        parameters
        ----------
            y_true : XXX
                true labels

            y_pred : XXX
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

        return mean_loss

    return binary_focal_loss_fixed
