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


class ArcMarginProduct(nn.Module):

    '''
    src: https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
    Additive Angular Margin Loss (or ArcFace)
    ArcFace, or Additive Angular Margin Loss, is a loss function used in face recognition tasks.
    The softmax is traditionally used in these tasks. 
    However, the softmax loss function does not explicitly optimise the feature embedding 
    to enforce higher similarity for intraclass samples and diversity for inter-class samples. 
    In other words, we want the ambeddings that are super similar to be VERY CLOSE to each-other 
    and the embeddings that are different to be VERY FAR from each-other
    '''


    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        '''
        in_features: dimension of the input
        out_features: dimension of the last layer (in our case the classification)
        s: norm of input feature
        m: margin
        ls_eps: label smoothing'''
        
        super(ArcMarginProduct, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # Fills the input `Tensor` with values according to the method described in
        # `Understanding the difficulty of training deep feedforward neural networks`
        # Glorot, X. & Bengio, Y. (2010)
        # using a uniform distribution.
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        one_hot = torch.zeros(cosine.size()).to(device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
