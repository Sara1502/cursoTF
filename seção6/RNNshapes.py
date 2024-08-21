import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


N = 1
T = 10
D = 3
K = 2
X = np.random.randn(N, T, D)

M = 5
i = tf.keras.layers.Input(shape=(T, D))
x = tf.keras.layers.SimpleRNN(M) (i)
x = tf.keras.layers.Dense(K) (x)

model = tf.keras.models.Model(i, x)

Yhats = model.predict(X)
print(Yhats)

model.summary()
model.layers[1].get_weights()

a, b, c = model.layers[1].get_weights()
print(a.shape, b.shape, c.shape)

Wx, Wh, bh = model.layers[1].get_weights()
Wo, bo = model.layers[2].get_weights()

h_last = np.zeros(M)
x = X[0]
Yhats = []

for t in range (T):
    h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh)
    Y = h.dot(Wo) + bo
    Yhats.append(Y)
    h_last = h

print(Yhats[-1])