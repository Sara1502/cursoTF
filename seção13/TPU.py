import tensorflow as tf
import numpy as np
import pandas as pd

from keras._tf_keras.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from keras._tf_keras.keras.models import Model



resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print('All devices: ', tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.TPUStrategy(resolver)


def create_model():
    i = Input(shape=(32, 32, 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same') (i)
    x = BatchNormalization() (x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = MaxPooling2D((2, 2)) (x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = MaxPooling2D((2, 2)) (x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = MaxPooling2D((2, 2)) (x)

    x = Flatten() (x)
    x = Dropout(0.2) (x)
    x = Dense(1024, activation='relu') (x)
    x = Dropout(0.2) (x)
    x = Dense(10) (x)

    model = Model(i, x)
    return model


cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
print("x_train.shape", x_train.shape) 
print("x_train.shape", y_train.shape)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategorical.crossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
batch_size = 256

train_dataset = train_dataset.shuffle(1000).batch(batch_size)
test_dataset = train_dataset.batch(batch_size)

model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset
)

model.save('mymodel.h5')

with strategy.scope():
    model = tf.keras.models.load_model('mymodel.h5')
    out = model.predict(x_test[:1])
    print(out)
