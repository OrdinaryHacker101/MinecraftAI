import tensorflow as tf
from tensorflow import keras

def build_encoder():
    enc = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size = 3, strides = 2, input_shape=[64,64,3]),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Conv2D(64, kernel_size = 3, strides = 1),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Conv2D(64, kernel_size = 3, strides = 2),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Conv2D(64, kernel_size = 3, strides = 1),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Conv2D(64, kernel_size = 3, strides = 2),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Reshape([8,8,64]),
        keras.layers.Dense(512)
    ])

    return enc
