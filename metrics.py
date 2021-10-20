import tensorflow.keras.backend as K
from tensorflow.keras.activations import sigmoid


def precision(y_true, y_pred):
    """
    computes batch-wise average of precision, metric for multi-label classification of how many selected items are relevant

    parameters
    ----------
        y_true : XXX
            XXX

        y_pred : XXX
            XXX

    returns
    -------
        precision : XXX
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
        y_true : XXX
            XXX

        y_pred : XXX
            XXX

    returns
    -------
        recall : XXX
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
        y_true : XXX
            XXX

        y_pred : XXX
            XXX

        apply_sigmoid_on_predicted_labels : boolean, default is False
            applies sigmoid function on predicted labels
            switch to True if dense layer is without activation function

    returns
    -------
        f1 : XXX
            value of f1 between true and predicted labels
    """

    if apply_sigmoid_on_predicted_labels:
        y_pred = sigmoid(y_pred)

    precisionScore = precision(y_true, y_pred)
    recallScore = recall(y_true, y_pred)

    f1 = 2 * ((precisionScore * recallScore) /
              (precisionScore + recallScore + K.epsilon()))

    return f1Score
