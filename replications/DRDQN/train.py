import gym

import numpy as np

from utils import ReplayBuffer

import tensorflow as tf
from tensorflow import keras

from PIL import Image

ITERATIONS = 100000

MINIBATCH_SIZE = 32
BUFFER_SIZE = 20000

LEARNING_RATE = 0.00025
TRAINING_FREQUENCY = 4
Y_UPDATE_FREQUENCY = 40000
UPDATE_FREQUENCY = 10000

REPLAY_START_SIZE = 50000

EPSILON_MAX = 1.0
EPSILON_MIN = 0.1
EPSILON_STEPS = 850000
DISCOUNT_FACTOR = 0.99

OPTIMIZER = keras.optimizers.RMSprop(lr=LEARNING_RATE)
LOSS_FN = keras.losses.mean_squared_error

env = gym.make("SpaceInvaders-v0")

IMG_SHAPE = (210, 160, 3)

N_ACTIONS = 6
print(N_ACTIONS)

replay_buffer = ReplayBuffer(BUFFER_SIZE, IMG_SHAPE)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=8, strides=4, input_shape=IMG_SHAPE),
    keras.layers.Conv2D(64, kernel_size=4, strides=2),
    keras.layers.Conv2D(64, kernel_size=3, strides=1),
    keras.layers.Flatten(),
    keras.layers.Reshape((1, 22528)),
    keras.layers.LSTM(512, return_sequences=True, activation="tanh"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(N_ACTIONS)
    ])

target = keras.models.clone_model(model)
target.set_weights(model.get_weights())

def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.save_mem(state, action, reward, next_state, done)
    return next_state, reward, done

def training_step(batch_size):
    experiences = replay_buffer.sample_mem(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = target.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, N_ACTIONS)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tf.reduce_mean(LOSS_FN(target_Q_values, Q_values))
    OPTIMIZER.apply_gradients(zip(grads, model.trainable_variables))

for episode in range(ITERATIONS):
    print("Episode: ",episode)
    
    done = False
    obs = env.reset()

    total_rewards = 0
    while not done:
        env.render(mode="rgb_array")

        #obs = np.array(Image.fromarray(obs, "RGB").convert("L"))

        epsilon = max(1 - episode / 500, 0.1)

        print(obs.shape)

        obs, reward, done = play_one_step(env, obs, epsilon)
        total_rewards += reward

    if episode > 50:
        training_step(MINIBATCH_SIZE)

    print("Total rewards: ", total_rewards)
        
        

    
