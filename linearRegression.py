import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('bancos/moore.csv', on_bad_lines='skip', header=None).to_numpy()

X= data[:,0].reshape(-1, 1)
Y = data[:,1]

plt.scatter(X, Y);
plt.show()

Y = np.log(Y)
plt.scatter(X, Y);
plt.show()

X = X - X.mean()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1, )),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimzer=tf.keras.optimizers.SGD(0.001, 0.9),
    loss='mse',
)

def scheduler(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

plt.plot(r.history['loss'], label='loss')
plt.legend()
plt.show()

model.layers
model.layers[0].get_weights()

a = model.layers[0].get_weights[0][0, 0]

print(np.log(2) / a)

Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat)

w, b = model.layers[0].get_weights()

X = X.reshape(-1, 1)

Yhat2 = (X.dot(w) + b).flatten()

np.allclose(Yhat, Yhat2)