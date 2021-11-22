from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from dataPreprocessing import datasets


pd.options.mode.chained_assignment = None  # default='warn'


# Graph train and test accuracy
def graphAccuracy(history):

    # Plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    # Plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()


# Confusion matrix
def confusionMatrix(y_test, y_pred):

    matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(matrix, columns=[0, 1], index = [0, 1])
    df_cm.index.name = 'Truth'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, cmap="BuGn", annot=True, annot_kws={"size": 16})
    plt.show()

    return matrix


# Print prediction metrics
def predictionMetrics(y_test, y_pred, y_class_pred, matrix):

    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]
    TN = matrix[0][0]

    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    g_mean = np.sqrt(sens * spec)

    accuracy = accuracy_score(y_test, y_class_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_class_pred)
    precision = precision_score(y_test, y_class_pred)
    recall = recall_score(y_test, y_class_pred)
    f1 = f1_score(y_test, y_class_pred)
    auc = roc_auc_score(y_test, y_pred)

    print('\t\t Prediction Metrics\n')
    print("Accuracy:\t", "{:0.3f}".format(accuracy))
    print("Precision:\t", "{:0.3f}".format(precision))
    print("Recall:\t\t", "{:0.3f}".format(recall))
    print("\nF1 Score:\t", "{:0.3f}".format(f1))
    print("ROC AUC:\t", "{:0.3f}".format(auc))
    print("Balanced\nAccuracy:\t", "{:0.3f}".format(balanced_accuracy))
    print("\nSensitivity:\t", "{:0.3f}".format(sens))
    print("Specificity:\t", "{:0.3f}".format(spec))
    print("Geometric Mean:\t", "{:0.3f}".format(g_mean))


def buildModel():

    # Data preparation
    _, _, _, _, n_features = datasets()

    # Architecture
    model = Sequential()
    model.add(Reshape((3197, 1), input_shape=(3197,)))
    model.add(Conv1D(filters=10, kernel_size=2, activation='relu', input_shape=(n_features, 1), kernel_regularizer='l2'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(48, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(18, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # Representation of architecture
    print(model.summary())

    return model

def trainModel(model, save_weights=False):

    # Data preparation
    x_train, y_train, x_test, y_test, _ = datasets()
    x_train, y_train = shuffle(x_train, y_train)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    model.fit(x_train, y_train, validation_split = 0.2, callbacks=[early_stop], batch_size=64, epochs=30, verbose=2)

    # Evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=2)
    _, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    # Prediction
    y_class_pred = (model.predict(x_test) > 0.5).astype("int32")
    y_pred = model.predict(x_test)

    # Confustion matrix
    conf_matrix = confusion_matrix(y_test, y_class_pred)

    # Metrics
    predictionMetrics(y_test, y_pred, y_class_pred, conf_matrix)

    if save_weights:

        # save best models
        result = []
        for i in range(1000):
            if conf_matrix[1][1] == 5:
                result.append(5)
                model.save('exoplanet_weights' + str(conf_matrix[0][1]) + 'i' + str(i) + '.h5')
            elif conf_matrix[1][1] == 4:
                result.append(4)
            elif conf_matrix[1][1] == 3:
                result.append(3)
            else:
                result.append(0)
        with open('models.npy', 'wb') as f:
            np.save(f, result)
