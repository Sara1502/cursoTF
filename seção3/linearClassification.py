import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = load_breast_cancer()

'''
print(type(data))
print(data.keys())
print(data.data.shape)
print(data.target)
print(data.target_names)
print(data.target.shape)
print(data.feature_names)'''

#Treinamento teste

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

N, D = X_train.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.models.Sequential ([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Alternativo
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(1, input_shape=(D, ), activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

print("Nota de treinamento: ", model.evaluate(X_train, y_train))
print("Nota do teste: ", model.evaluate(X_test, y_test))
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend();
plt.show()

p = model.predict(X_test)
print(p)

p = np.round(p).flatten()

print(p)
print("Precisão calculada manualmente: ", np.mean(p == y_test))
print("Avaliação do output: ", model.evaluate(X_test, y_test))