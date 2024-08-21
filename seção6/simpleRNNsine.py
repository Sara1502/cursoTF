import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


series = np.sin(0.1*np.arange(200)) + np.random.random(200)*0.1

plt.plot(series)
plt.show()

T = 10
D = 1
X = []
Y = []
for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y)
N = len(X)
print('X.shape ', X.shape, 'Y.shape ', Y.shape)

#Autoregressive RNN model
i = tf.keras.layers.Input(shape=(T, 1))
x = tf.keras.layers.SimpleRNN(5, activation='relu') (i)
x = tf.keras.layers.Dense(1) (x)
model = tf.keras.models.Model(i, x)
model.compile(
    loss='mse',
    optimizer = tf.keras.optimizers.Adam(lr=0.1),
)

r = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs=80,
    validation_data=(X[-N//2:], Y[-N//2:]),
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

validation_target = Y[-N//2:]
validation_pretctions = []

last_x = X[-N//2]

while len(validation_pretctions) < len(validation_target):
    p = model.predict(last_x.reshape(1, -1)) [0, 0]

    validation_target.append(p)

    last_x = np.roll(last_x, -1)
    last_x[-1] = p

plt.plot(validation_target, label='foscast target')
plt.plot(validation_pretctions, label='forcast prediction')
plt.legend()
plt.show()