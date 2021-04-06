from utils import *

import gym
import minerl

import numpy as np
from collections import deque

from PIL import Image

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

#NOTE: FOR NOW THIS IS ONLY A BASIC DEEP Q LEARNING ALGORITHM. NO LEARNING FROM VIDEO
#HAS BEEN APPLIED YET TO THIS CODE.

#ALL CREDITS GOES TO:
#VCode1423
#kimbring2
#thomas simnonini
#Aurelien Geron's Hands on ML

#WITHOUT ANY OF THESE PEOPLE THIS WOULD HAVE NEVER BEEN POSSIBLE, SO I THANK THEM ALL
#PERSONALLY.

action_labels = ["forward", "left", "back", "right", "jump", "sneak", "sprint", "attack", "place", "camera"]
obs_metalabels = ["compassAngle", "inventory", "pov"]

action_range = 11

#TRAINING_HYPERPARAMETERS
EPISODES = 100
MAX_STEPS = 1000
DISCOUNT_FACTOR = 0.6
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.95

OPTIMIZER = keras.optimizers.Adam(lr=1e-3)
LOSS_FN = keras.losses.mean_squared_error

#frame shapes
stacked_frame_shape = [64,64,5]
stack_size = 4

#chkpt directory
chkpt_dir = "C:\\Users\\Robin\\Desktop\\deep_learning\\DQFD\\chkpts"

#model necessities
model = build_dqn()
memory = ReplayBuffer(20000, stacked_frame_shape)

def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(action_range)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def sample_experiences(batch_size):
    states, n_states, actions, rewards, dones = memory.sample_mem(batch_size)
    return states, n_states, actions, rewards, dones

def play_one_step(env, state, epsilon, stacked_frames, episode_start):
    
    action_index = epsilon_greedy_policy(state, epsilon)
    action = env.action_space.noop()
    
    if action_index < 9:
        action[action_labels[action_index]] = 1
    elif action_index == 9:
        action["camera"] = [0, -5]
    elif action_index == 10:
        action["camera"] = [0, 5]
        
    n_state, reward, done, info = env.step(action)
    n_state, stacked = converter(n_state, stacked_frames, episode_start)

    #to be changed when HumanDQN is applied
    memory.save_mem(state, action_index, reward, n_state, done)

    return n_state, reward, done, stacked

def training_step(batch_size):
    experiences = memory.sample_mem(batch_size)
    states, n_states, actions, rewards, dones = experiences
    next_Q_values = model.predict(n_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * DISCOUNT_FACTOR * max_next_Q_values)
    mask = tf.one_hot(actions, action_range)
    with tf.GradientTape() as Tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis = 1, keep_dims=True)
        loss = tf.reduce_mean(LOSS_FN(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train_model():
    env = gym.make("MineRLNavigateDense-v0")
    stacked_frames = deque([np.zeros((64,64), dtype=np.uint) for i in range(stack_size)], maxlen=4)

    for episode in range(EPISODES):
        print("Episode: ", episode)

        obs = env.reset()
        episode_start = True
        done = False
        
        total_rewards = 0
        obs, to_be_stacked = converter(obs, stacked_frames, episode_start)

        stacked_frames = to_be_stacked
        
        for step in range(MAX_STEPS):
            img = env.render(mode="rgb_array")
            epsilon = max(1 - episode/ 500, 0.01)
            print(obs)
            
            obs, reward, done, to_be_stacked = play_one_step(env, obs, epsilon, stacked_frames, episode_start)

            total_rewards += reward
            if done:
                break

            episode_start = False
            
        if episode > 50:
            training_step(BATCH_SIZE)

        print("Total rewards: ", total_rewards)

if __name__ == "__main__":
    train_model()
        
        
