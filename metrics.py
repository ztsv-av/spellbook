import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.activations import sigmoid


def precision(y_true, y_pred):
    """
    computes batch-wise average of precision, metric for multi-label classification of how many selected items are relevant

    parameters
    ----------
        y_true : tensor
            true labels

        y_pred : tensor
            predicted labels

    returns
    -------
        precision : number
            value of precision between true and predicted labels
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision


def recall(y_true, y_pred):
    """
    computes batch-wise average of recall, metric for multi-label classification of how many relevant items are selected

    parameters
    ----------
        y_true : tensor
            true labels

        y_pred : tensor
            predicted labels

    returns
    -------
        recall : number
            value of recall between true and predicted labels
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall


def f1(y_true, y_pred, apply_sigmoid_on_predicted_labels=False):
    """
    computes batch-wise average of f1, metric which is a combination of precision and recall

    parameters
    ----------
        y_true : tensor
            true labels

        y_pred : tensor
            predicted labels

        apply_sigmoid_on_predicted_labels : boolean, default is False
            applies sigmoid function on predicted labels
            switch to True if dense layer is without activation function

    returns
    -------
        f1 : number
            value of f1 between true and predicted labels
    """

    if apply_sigmoid_on_predicted_labels:
        y_pred = sigmoid(y_pred)

    precisionScore = precision(y_true, y_pred)
    recallScore = recall(y_true, y_pred)

    f1 = 2 * ((precisionScore * recallScore) /
              (precisionScore + recallScore + K.epsilon()))

    return f1


def map5PerExample(y_true, y_pred):
    """
    computes the precision score of one example

    parameters
    ----------
    y_true : int/double/string/anything
            true label of the  example (one true element)

    y_pred : list
            list of predicted elements

    returns
    -------
        score : double
            map5 score for one example
    """

    try:
        score = 1 / (y_pred[:5].index(y_true) + 1)
        return score

    except ValueError:
        return 0.0


def map5(y_true, y_pred):
    """
    computes the average over multiple examples (batch)

    parameters
    ----------
    y_true : list
             list of the true labels. (only one true label per example allowed!)

    y_pred : list of list
             list of predicted elements (order does matter, 5 predictions allowed per example)

    example : 
        y_true = [1, 2, 3, 4, 5]
        y_pred = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        b = [map5PerExample(l, p) for l,p in zip(y_true, y_pred)] => [1.0, 0.5, 0.333, 0.25, 0.2]
        a = map5(y_true, y_pred) => 0.458

    returns
    -------
    score : double
        map5 metric value for a whole batch
    """

    score = np.mean([map5PerExample(l, p) for l,p in zip(y_true, y_pred)])

    return score
