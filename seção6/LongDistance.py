import tensorflow as tf
from keras._tf_keras.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, GlobalMaxPool1D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


T = 10
D = 1
X = []
Y = []

def get_label(x, i1, i2, i3):
    if x[i1] < 0 and x[i2] < 0 and x[i3] < 0:
        return 1
    if x[i1] < 0 and x[i2] > 0 and x[i3] > 0:
        return 1
    if x[i1] > 0 and x[i2] < 0 and x[i3] > 0:
        return 1
    if x[i1] > 0 and x[i2] > 0 and x[i3] < 0:
        return 1
    return 0

for t in range(5000):
    x = np.random.randn(T)
    X.append(x)
    #y = get_label(x, -1, -2, -3) #Distancia curta
    y = get_label(x, -1, -2, -3) #Distancia longa
    Y.append(y)

X = np.array(X)
Y = np.array(Y)
N = len(X)

i = Input(shape=(T, ))
x = Dense(1, activation='sigmoid') (i)
model = Model(i, x)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy'],
)

r = model.fit(
    X, Y,
    epochs=100,
    validation_split=0.5
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

#SimpleRNN
inputs = np.expand_dims(X, -1)
i = Input(shape=(T, D))

x = SimpleRNN(5) (i)
x = Dense(1, activation='sigmoid') (x)
model = Model(i, x)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy'],
)

r = model.fit(
    inputs, Y,
    epochs=200,
    validation_split=0.5
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()


#LSTM
inputs = np.expand_dims(X, -1)
i = Input(shape=(T, D))

x = LSTM(5) (i)
x = Dense(1, activation='sigmoid') (x)
model = Model(i, x)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy'],
)

r = model.fit(
    inputs, Y,
    epochs=200,
    validation_split=0.5
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

#Mais facil
inputs = np.expand_dims(X, -1)
i = Input(shape=(T, D))

x = GlobalMaxPool1D(5) (i)
x = Dense(1, activation='sigmoid') (x)
model = Model(i, x)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy'],
)

r = model.fit(
    inputs, Y,
    epochs=100,
    validation_split=0.5
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()