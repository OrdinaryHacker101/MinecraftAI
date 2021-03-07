import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np
import time

import matplotlib.pyplot as plt
import tqdm

#AZNET
model = keras.models.load_model("C:\\Users\\Robin\\Desktop\\deep_learning\\models\\AZNET.h5")

class DDDQN(keras.Model):
    def __init__(self, n_actions, main_model, fc1_dims, fc2_dims):
        super(DDDQN, self).__init__()
        self.main_model = main_model
        #lays = main_model.layers
        #self.the_layers = lays.set_weights(main_model.get_weights())
        #self.the_layers.set_weights(self.main_model.get_weights())
        #self.dense1 = keras.layers.Dense(fc1_dims, activation="elu")
        #self.dense2 = keras.layers.Dense(fc2_dims, activation="elu")
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        A = self.main_model(state)
        self.main_model.summary()
        #x = self.dense1(x)
        #x = self.dense2(x)
        V = self.V(A)
        #A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    #def mouse(self, state):
        #M = self.main_model(state)
        #x = self.dense1(x)
        #x = self.dense2(x)
        #A = self.A(x)

        #return M

    def advantage(self, state):
        x = self.main_model(state)
        A = self.A(x)

        return A

class ReplayBuffer():

    def __init__(self, max_size, input_shape, obs_size=[3]):
        self.mem_size = max_size
        self.img_size = [64, 64, 3]
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *self.img_size),
                                     dtype=np.float32)
        self.state_val = np.zeros((self.mem_size, *obs_size), dtype=np.float32)
        
        self.new_state_memory = np.zeros((self.mem_size, *self.img_size),
                                         dtype=np.float32)
        self.new_state_val = np.zeros((self.mem_size, *obs_size), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, s_val, action, reward, state_, s_val_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.state_val[index] = s_val
        self.new_state_memory[index] = state_
        self.new_state_val[index] = s_val_
        self.action_memory[index] = action 
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        state_vals = self.state_val[batch]
        states_ = self.new_state_memory[batch]
        state_vals_ = self.new_state_val[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, state_vals, actions, rewards, states_, state_vals_, dones

class Agent():
    #this class combines the models, memories, and hyperparameters that everything
    #is training on.
    def __init__(self, lr, gamma, n_actions, main_model, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, eps_end=0.01,
                 mem_size=1000, fc1_dims=128, fc2_dims=128, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DDDQN(n_actions, main_model, fc1_dims, fc2_dims)
        self.q_next = DDDQN(n_actions, main_model, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss="mean_squared_error")
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss="mean_squared_error")

    def store_transition(self, state, s_val, action, reward, state_, s_val_, done):
        self.memory.store_transition(state, s_val, action, reward, state_, s_val_, done)

    #chooses if the action is random or calculated by the model. It starts by
    #having a random action moved, but as time goes on, the self.epsilon variable
    #decreases, and it starts to use the model more and more.
    
    def choose_action(self, observation):
        #def mouse_control(observation):
            #state = observation
            #q_eval = DDDQN(8, model, 1, 2)
            #preds = q_eval.mouse(state)[0]
            #camera_x, camera_y = preds[-2], preds[-1]
            #return camera_x, camera_y
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            camera_x, camera_y = np.random.randint(-180, 181, 2)
        else:
            state = observation
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
            #camera_x, camera_y = mouse_control(state)

        return action, [0, 0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        print(self.learn_step_counter%self.replace)
        if self.learn_step_counter % self.replace == 0:
            print(np.asarray(self.q_eval.get_weights()).shape)
            self.q_next.set_weights(np.asarray(self.q_eval.get_weights()))

        #state, s_val, action, reward, state_, s_val_, done
        states, s_vals, actions, rewards, states_, s_vals_, dones = \
                self.memory.sample_buffer(self.batch_size)
        states = [states, s_vals]
        states_ = [states_, s_vals_]
        #predicted current q values
        q_pred = self.q_eval(states)
        
        #predicted target q values
        q_next = self.q_next(states_)
        # changing q_pred doesn't matter because we are passing states to the train
        # function anyway.
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)

        # improve this part
        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min
        self.learn_step_counter += 1
        
class Policy_Gradients():
    pass
