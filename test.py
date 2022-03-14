from re import X
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import keras
plt.style.use('ggplot')
import datetime
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

np.seterr(divide='ignore', invalid='ignore')
# labels = [["sastramelayu","sastrajawa","english","non_labels"]]
labels = [["A","B","C"]]
mlb = MultiLabelBinarizer()
mlb.fit(labels)
MultiLabelBinarizer(classes=None, sparse_output=False)


y_true = ['A', 'B', 'C', 'B', 'A', 'C', 'C', 'B', 'A', 'C']

y_pred = ['A', 'B', 'C', 'B', 'A', 'C', 'C', 'B', 'A', 'C']

#A = 1
#B = 0.5
#C = 0.8
weight = []

y_true_transform = []




def weighted_f1(y_true, y_pred, weight):
    for x in range(0, len(y_true)):
        # print(y_true[x])
        y_tmp = mlb.transform([[y_true[x]]])
        # print(len(y_tmp))
        y_true_transform.append(y_tmp[0])

    print(y_true_transform)


    precission = (weight * np.array(y_true_transform)) / (weight * np.array(y_true_transform))
    recall = (weight * np.array(y_true_transform))  / (weight * np.array(y_true_transform))
    f1_weight = 2 * (precission * recall / (precission + recall))

    print(precission)
    print(recall)
    print(f1_weight)

    return precission, recall, f1_weight

weighted_f1(y_true, y_pred, [0.5])