import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.models import load_model
from itertools import chain

from dataPreprocessing import datasets
from train import predictionMetrics

pd.options.mode.chained_assignment = None  # default='warn'


model = load_model('perfect_weights.h5')

_, _, x_test, y_test, _ = datasets()
y_class_pred = (model.predict(x_test) > 0.5).astype("int32")
y_pred = model.predict(x_test)

y_test = y_test.tolist()
y_class_pred = y_class_pred.tolist()
y_class_pred = list(chain.from_iterable(y_class_pred))

matrix = confusion_matrix(y_test, y_class_pred)
predictionMetrics(y_test, y_pred, y_class_pred, matrix)
