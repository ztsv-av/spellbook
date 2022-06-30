import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.activations import sigmoid

from globalVariables import FROM_LOGITS

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


def f1Wrapper():
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

    def f1(y_true, y_pred):

        if FROM_LOGITS:
            y_pred = sigmoid(y_pred)

        precision_score = precision(y_true, y_pred)
        recall_score = recall(y_true, y_pred)

        f1_score = 2 * ((precision_score * recall_score) /
                (precision_score + recall_score + K.epsilon()))

        return f1_score
    
    return f1


def map5Wrapper():
    """
    computes the average over multiple examples (batch)

    parameters
    ----------
    y_true : list of tensors
             batch of tensors of the true labels, one-hot encoded (only one true label per example allowed!)

    y_pred : list of tensors
             batch of tensors of predicted elements

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

    def map5(y_true, y_pred):

        y_true_top_k = tf.math.top_k(y_true, k=1, sorted=True, name=None)
        y_true_top_k_idxs = y_true_top_k.indices
        y_true_top_k_idxs = tf.reshape(y_true_top_k_idxs, (y_true_top_k_idxs.shape[0], 1))

        y_pred_top_k = tf.math.top_k(y_pred, k=5, sorted=True, name=None)
        y_pred_top_k_idxs = y_pred_top_k.indices

        scores = tf.where(tf.equal(y_pred_top_k_idxs, y_true_top_k_idxs))[:,-1]

        is_empty = tf.equal(tf.size(scores), 0)
        
        if not is_empty:

            scores = 1 / (tf.add(scores, 1))

            score = tf.reduce_mean(scores)

        else:

            score = tf.cast(0, dtype=tf.float64)

        return score

    return map5
