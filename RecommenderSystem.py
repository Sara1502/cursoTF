import tensorflow as tf
from keras._tf_keras.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import SGD, Adam

from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('ml-20m/ratings.csv')
print(df.head())

df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes

df.movieId = pd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes

user_ids = df['new_user_id'].values
movies_ids = df['new_movie_id'].values
ratings = df['rating'].values

M = len(set(user_ids))
N = len(set(movies_ids))

K = 10
#user input
u = Input(shape=(1,))
#movie input
m = Input(shape=(1,))

u_emb = Embedding(N, K) (u)
m_emb = Embedding(N, K) (m)

u_emb = Flatten() (u_emb)
m_emb = Flatten() (m_emb)

x = Concatenate() ([u_emb, m_emb])
x = Dense(1024, activation='relu') (x)
x = Dense(1) (x)

model = Model(inputs=[u, m], outputs=x)
model.compile(
    loss='mse',
    optimizer=SGD(lr=0.08, momentum=0.9),
)

user_ids, movies_ids, ratings = shuffle(user_ids, movies_ids, ratings)
Ntrain = int(0.8 * len(ratings))
train_user = user_ids[:Ntrain]
train_movie = movies_ids[:Ntrain]
train_ratings = ratings[:Ntrain]

test_user = user_ids[Ntrain:]
test_movie = movies_ids[Ntrain:]
test_ratings = ratings[Ntrain:]

avg_rating = train_ratings.mean()
train_ratings = movies_ids - avg_rating
test_ratings = test_ratings - avg_rating

r = model.fit(
    x=[train_user, train_movie],
    y=train_ratings,
    epochs=25,
    bacht_size = 1024,
    verbose=2,
    validation_data=([test_user, test_movie], test_ratings),
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()