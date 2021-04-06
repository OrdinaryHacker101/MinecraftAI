import numpy as np
from collections import deque

from PIL import Image

import tensorflow as tf
from tensorflow import keras

stack_size = 4

def stack_frames(stacked_frames, state, episode_start):

    frame = state

    if episode_start:
        
        stacked_frames = deque([np.zeros((64,64), dtype=np.uint) for i in range(stack_size)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(list(stacked_frames), axis=2)

    else:

        stacked_frames.append(frame)
        stacked_state = np.stack(list(stacked_frames), axis=2)
        
        
    return stacked_state, stacked_frames

def converter(observation, stacked_frames, episode_start):
    obs = list(observation.items())[2][1]
    obs = np.array(Image.fromarray(obs, "RGB").convert("L"))
    obs = obs / 255
    obs, stacked_frames = stack_frames(stacked_frames, obs, episode_start)
    
    print(obs.shape)
                                                   
    compass_angle = list(observation.items())[0][1]
    print(compass_angle)

    compass_angle_scale = 180
    compass_scaled = compass_angle / compass_angle_scale
    compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
    obs = np.concatenate([obs, compass_channel], axis=-1)

    return obs, stacked_frames

def build_dqn():
    dqn = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=5, input_shape=[64, 64, 5]),
        keras.layers.Conv2D(32, kernel_size=3),
        keras.layers.Flatten(),
        keras.layers.Dense(11)
        
    ])

    return dqn

class ReplayBuffer():
    def __init__(self, mem_size, img_shape):
        self.mem_cntr = 0
        self.mem_size = mem_size
        self.img_shape = img_shape

        #MEMORIES
        self.state_mem = np.zeros((self.mem_size, *img_shape),
                                  dtype = np.float32)
        self.n_state_mem = np.zeros((self.mem_size, *img_shape),
                                    dtype = np.float32)
        self.action_mem = np.zeros(self.mem_size,
                                   dtype = np.float32)
        self.reward_mem = np.zeros(self.mem_size, dtype = np.float32)
        self.done_mem = np.zeros(self.mem_size, dtype = np.bool)

    def save_mem(self, state, action, reward, n_state, done):

        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.n_state_mem[index] = n_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.done_mem = done

        self.mem_cntr += 1

    def sample_mem(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[batch]
        n_states = self.n_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.done_mem[batch]

        return states, n_states, actions, rewards, dones
        
        
