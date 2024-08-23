import tensorflow as tf
from keras._tf_keras.keras.layers import Input, Dense, GlobalMaxPooling1D
from keras._tf_keras.keras.layers import Conv1D, MaxPooling1D, Embedding
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('bancos/spam.csv', encoding='ISO-8859-1')
print(df.head())

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
print(df.head())

df.columns = ['labels', 'data']
print(df.head())

df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
Y = df['b_labels'].values

df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)

MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)

word2idx = tokenizer.word_index
V = len(word2idx)
print('Found %s unique tokens.' % V)

data_train = pad_sequences(sequences_train)
print('Shape of data train tensor: ', data_train.shape)

T = data_train.shape[1]

data_test = pad_sequences(sequences_test, maxlen=T)
print('Shape of data test tensor: ', data_test.shape)

#Criar modelo
D = 20

i = Input(shape=(T, ))
x = Embedding(V + 1, D) (i)
x = Conv1D(32, 3, activation='relu') (x)
x = MaxPooling1D(3) (x)
x = Conv1D(64, 3, activation='relu') (x)
x = MaxPooling1D(3) (x)
x = Conv1D(128, 3, activation='relu') (x)
x = GlobalMaxPooling1D() (x)
x = Dense(1, activation='sigmoid') (x)

model = Model(i, x)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print('Training Model...')
r = model.fit(
    data_train,
    Ytrain,
    epochs=5    ,
    validation_data=(data_test, Ytest)
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()