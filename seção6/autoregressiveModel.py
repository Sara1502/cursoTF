import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

#Data origial
series = np.sin(0.1*np.arange(200))

plt.plot(series)
plt.show()

#Construindo o dataset
T = 10
X = []
Y = []
for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)

print('X.shape: ', X.shape, 'Y.shape: ', Y.shape)

#Autoresgressive linear model
i = tf.keras.layers.Input(shape=(T, ))
x = tf.keras.layers.Dense(1) (i)
model = tf.keras.models.Model(i, x)
model.compile(
    loss='mse',
    optimizer = tf.keras.optimizers.Adam(lr=0.1),
)

#Treinar RNN
r = model.fit(X[:-N//2], Y[:-N//2],
              epoch = 80,
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