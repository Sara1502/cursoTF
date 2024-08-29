import numpy as np
import pandas as pd

import tensorflow as tf
from keras._tf_keras.keras.layers import Input, Dense
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

# AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
    # return T x 3 list of stock prices
    # each row is a different stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    df = pd.read_csv('bancos/aapl_msi_sbux.csv')
    return df.values

### The experience replay memory ###
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size, = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obe2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews.buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])
    


def get_scaler(env):
    # return scikit-learn scaler object to scale the states

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

        scaler = StandardScaler()
        scaler.fit(states)
        return scaler
    


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):

    # input layer
    i = input(shape=(input_dim,))
    x = i

    # hidden layer
    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation='relu') (x)
    
    # final layer
    model = Model(i, x)

    model.compile(loss='mse', optimizer='adam')
    print((model.summary()))
    return model


class MultiStockEnv:
    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape
        
        # instance attributes
        self.initial_investment  = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3**self.n_stock)

        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        self.state_dim = self.n_stock * 2 + 1

        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()
    

    def step(self, action):
        assert action in self.action_space

        prev_val = self._get_val()

        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        self._trade(action)

        cur_val = self._get_val()

        reward = cur_val - prev_val

        done = self.cur_step == self.n_step - 1

        info = {'cur_val': cur_val}

        return self._get_obs(), reward, done, info
    

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs
    

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand
    

    def _tarde(self, action):
        action_vec = self.action_list[action]

        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)


        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False



class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95 # discout rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)

    
    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_value = self.model.predict(state)
        return np.argmax(act_value[0])
    

    def replay(self, batch_size=32):
        if self.memory.size < batch_size:
            return
        
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)

        target[done] = rewards[done]


        target_full  = self.model.predict(states)
        target_full[np.arange(batch_size), actions] = target


        self.model.train_on_batch(states, target_full)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
            self.model.load_weights(name)

        
    def save(self, name):
        self.model.load_weightd(name)
