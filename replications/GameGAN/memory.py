#Needs 3 mlps: K, G, and E
#All memory is stored in memory box M
#K produces 3x3 kernels
#G outputs a scalar value
#E produces erase and add vectors
import numpy as np

import tensorflow as tf
from tensorflow import keras

class Memory():
    def __init__(selfl, max_size, obs_size=[3]):
        self.mem_size = max_size
        self.state_mem = np.zeros((self.mem_size, *[512]),
                                  dtype=np.float32)

        self.K = keras.models.Sequential([
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(9),
            keras.layers.Softmax(),
            keras.layers.Reshape(3,3)
        ])

        self.G = keras.models.Sequential([
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(1),
            keras.layers.Sigmoid()
        ])

        self.E = keras.models.Sequential([
            keras.layers.Dense(1024),
            tf.split(num_or_size_splits = 2)
        ])
        
