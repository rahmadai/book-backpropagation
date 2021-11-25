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
import numpy as np


file_dataset = {'sastra': 'dataset/dataset_full.csv'}

df_list = []
for source, filepath in file_dataset.items():
    # No|Judul|Penulis|Genre|Tahun Terbit|Kategori
    df = pd.read_csv(filepath, names=['No','Judul','Penulis','Genre','Tahun Terbit','Kategori'], sep='|')

df_sastra = df
sentences = df_sastra['Judul'].values
y = df_sastra['Kategori'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=355)



tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
maxlen = 100

X_infer = tokenizer.texts_to_sequences(['ngopo koe ngalih wae'])
# print(X_infer)
X_infer = pad_sequences(X_infer, padding='post', maxlen=maxlen)

#[["melayu","jawa","english","non_labels"]]
#["english", "jawa", "melayu", "non_labels"]
#["english", "jawa", "melayu"]

model = keras.models.load_model('model_backprop_without_non_labels')
predict_x=model.predict(X_infer) 
classes_x=np.argmax(predict_x,axis=1)
predict_y = ""

class_disb = np.argmax(predict_x)
if class_disb == 0:
    predict_y = "english"
elif class_disb == 1:
    predict_y = "sastrajawa"
elif class_disb == 2:
    predict_y = "sastramelayu"
elif class_disb == 3:
    predict_y = "non_labels"

print(predict_x, np.argmax(predict_x), predict_y)

# new_complaint = ['matamu neng gusti']
# seq = tokenizer.texts_to_sequences(new_complaint)
# padded = pad_sequences(seq, maxlen=maxlen)
# pred = model.predict(np.array(padded))
# print(pred, np.argmax(pred))