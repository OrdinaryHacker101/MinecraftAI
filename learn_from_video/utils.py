import numpy as np
from collections import deque

from PIL import Image

from trees import *

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
        self.done_mem[index] = done

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

        
class PER():

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4

    PER_b_increment_per_sampling = 0.001

    def __init__(self, capacity):

        self.tree = SumTree(capacity)
        self.absolute_error_upper = 1.

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):

            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)
            sampling_probabilities = priority / self.tree.total_priority

            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
            b_idx[i] = index

            experience = [data]
            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        print(ps)
        for ti, p in zip(tree_idx, ps):
            print(ti, p)
            p = tf.reduce_mean(p)
            self.tree.update(ti, p)

            
