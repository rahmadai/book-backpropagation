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

# labels = [["sastramelayu","sastrajawa","english","non_labels"]]
labels = [["sastramelayu","sastrajawa","english"]]
mlb = MultiLabelBinarizer()
mlb.fit(labels)
MultiLabelBinarizer(classes=None, sparse_output=False)



file_dataset = {'sastra': '/home/server/rahmad/Work/Other/Skripsi/dataset/dataset_full.csv'}

df_list = []
for source, filepath in file_dataset.items():
    # No|Judul|Penulis|Genre|Tahun Terbit|Kategori
    df = pd.read_csv(filepath, names=['No','Judul','Penulis','Genre','Tahun Terbit','Kategori'], sep='|')

df_sastra = df
sentences = df_sastra['Judul'].values
y = df_sastra['Kategori'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=355)

# y_test = mlb.transform([[y_test]])
y_train_transform = []
y_test_transform = []



for x in range(0, len(y_test)):
    print(y_test[x])
    y_tmp = mlb.transform([[y_test[x]]])
    print(len(y_tmp))
    y_test_transform.append(y_tmp[0])

for x in range(0, len(y_train)):
    print(y_train[x])
    y_tmp = mlb.transform([[y_train[x]]])
    print(len(y_tmp))
    y_train_transform.append(y_tmp[0])

# print(y_train_transform)
# print(y_test_transform)

# print("sentences_test = ")
# print(sentences_test)
# print("y_test = ")
# print(y_test)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
print(len(X_train))
X_test = tokenizer.texts_to_sequences(sentences_test)
print(len(X_test))

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print("vocab_size :")
print( vocab_size)

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

log_dir = "logs/fit/" + "melayujawa"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(3, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(np.array(X_train), np.array(y_train_transform),
                    epochs=50,
                    verbose=True,
                    validation_data=(np.array(X_test), np.array(y_test_transform)),
                    batch_size=10,
                    callbacks=[tensorboard_callback])
loss, accuracy = model.evaluate(np.array(X_train), np.array(y_train_transform), verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
val_loss, val_accuracy = model.evaluate(np.array(X_test), np.array(y_test_transform), verbose=False)
print("Testing Accuracy:  {:.4f}".format(val_accuracy))

print(X_test)
model.save("model_backprop_without_non_labels")