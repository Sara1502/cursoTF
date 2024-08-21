import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


series = np.sin((0.1*np.arange(400))**2)

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

X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print('X.shape ', X.shape, 'Y.shape ', Y.shape)

i = tf.keras.layers.Input(shape=(T, 1))
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

outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:, 0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title('Linear Regression Predictions')
plt.legend()
plt.show()

#Com RNN/LSTM model
X = X.reshape(-1, T, 1)

i = tf.keras.layers.Input(shape=(T, D))
x = tf.keras.layers.SimpleRNN(10) (i)
x = tf.keras.layers.Dense(1) (x)
model = tf.keras.models.Model(i, x)
model.compile(
    loss='mse',
    optimizer = tf.keras.optimizers.Adam(lr=0.05),
)

r = model.fit(
    X[:-N//2], Y[:-N//2],
    batch_size=32,
    epochs=200,
    validation_data=(X[-N//2:], Y[-N//2:]),
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title('many-to-one RNN')
plt.legend()
plt.show()

forecast = []
input_ = X[-N//2]

while len(forecast) < len(Y[-N//2:]):
    f = model.predict(input_.reshape(1, T, 1)) [0, 0]
    forecast.append(f)

    input_ = np.roll(input_, -1)
    input_[-1] = f

plt.plot(Y[-N//2:], label='targets')
plt.plot(forecast, label='forecast')
plt.title('RNN Forecast')
plt.legend()
plt.show()

