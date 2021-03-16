# Needs a dynamics engine, memory module, and a rendering engine
# Trains off of human gameplay (keyboard and frames)

#### DYNAMICS ENGINE ####
# vt(state value) = ht-1(previous hidden state) hadamard multiply MLP(action, stochastic variable, mt-1(previous memory))
# st(state encodings) = CNNencoder(xt(image at time))
# it(input gate) = sigmoid(Weights of vt's input * vt + weights of st's input* st)
# ft(forget gate) = sigmoid(Weights of vt's forget * vt + weights of st's forget* st)
# ot(output gate) = sigmoid(Weights of vt's output * vt + weights of st's output* st)
# ct(cell state) = ft hadamard multiply tanh(Weights of vt's cell state * vt + weights of st's cell state * st)
# ht(hidden state) = ot hadamard multiply tanh(ct)

import numpy as np

import tensorflow as tf
from tensorflow import keras

from model_utils import build_encoder

from memory import Memory

#class BuildModels():
def build_mlp():
    H = keras.models.Sequential([
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(512)
    ])
        
    return H

class ActionLSTM(keras.Model):
    def __init__(self, input_dim, hidden_size = 1024, opts = None):
        super(ActionLSTM, self).__init__()
        self.opts = opts
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.v_size = hidden_size
        self.project_input = keras.layers.Dense(self.input_dim)
        self.project_h = keras.layers.Dense(self.input_dim)
        

        self.v2h = keras.models.Sequential([
            keras.layers.Dense(self.input_dim),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(self.hidden_size)
        ])

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.get_weights():
            w.set_weights(np.random.uniform(-std, std))

    def init_hidden(self, bs):

        return np.zeros(bs, self.hidden_size)
    
    def forward(self, h, c, INPUT, state_bias = None, step = 0):
        
        h_proj = self.project_h(h)

        input_proj = self.project_input(INPUT)

        v = self.v2h(h_proj * input_proj)
        print(v)

        if state_bias is None:
            state_bias = 0
        total_w = v + state_bias
        
        g_t = keras.activations.tanh(total_w[:, 2 * self.hidden_size:3 * self.hidden_size])
        i_t = keras.activations.sigmoid(total_w[:, :self.hidden_size])
        f_t = keras.activations.sigmoid(total_w[:, self.hidden_size:2 * self.hidden_size])
        o_t = keras.activations.sigmoid(total_w[:, -self.hidden_size:])
        
        c_t = tf.math.multiply(c, f_t) + tf.math.multiply(i_t, g_t)
        h_t = tf.math.multiply(o_t, keras.activations.tanh(c_t))
        
        return h_t, c_t

class EngineModule():
    def __init__(self, opts, state_dim):
        super(EngineModule, self).__init__()
        self.hdim = opts.hidden_dim
        self.opts = opts

        action_space = 10

        self.a_to_input = keras.models.Sequential([
            keras.layers.Dense(self.hdim),
            keras.layers.LeakyReLU(0.2)
        ])

        self.z_to_input = keras.models.Sequential([
            keras.layers.Dense(self.hdim),
            keras.layers.LeakyReLU(0.2)
        ])

        e_input_dim = self.hdim*2
        if self.opts.do_memory:
            e_input_dim += self.opts.memory_dim

        self.f_e = keras.models.Sequential([
            keras.layers.Dense(e_input_dim),
            keras.layers.LeakyReLU(0.2)
        ])

        self.rnn_e = ActionLSTM(e_input_dim,
                                hidden_size=self.hdim, opts=opts)
        
        self.state_bias = keras.models.Sequential([
            keras.layers.Dense(self.hdim*4)
        ])

    def init_hidden(self, bs):
        return np.zeros(bs, self.rnn_e.hidden_size)

    def forward(self, h, c, s, a, z, prev_read_v=None, step=0):
        h_e = h[0]
        c_e = c[0]

        if self.opts.do_memory:
            input_core = [self.a_to_input(a), self.z_to_input(z), prev_read_v]
        else:
            input_core = [self.a_to_input(a), self.z_to_input(z)]
        state_bias = self.state_bias(s)
        input_core = np.concatenate(input_core, axis=1)

        e = self.f_e(input_core)
        h_e_t, c_e_t = self.rnn_e(h_e, c_e, e, state_bias=state_bias, step=step)

        H = [h_e_t]
        C = [c_e_t]

        return H, C, h_e_t

class EngineGenerator():

    def __init__(self, opts, nfilter_max=512, **kwargs):
        super(EngineGenerator, self).__init__()

        self.opts = opts

        self.z_dim = opts.z
        self.num_components = opts.num_components
        self.hdim = opts.hidden_dim
        self.expand_dim = opts.nfilterG

        if opts.do_memory:
            self.base_dim = opts.memory_dim
        else:
            self.base_dim = opts.hidden_dim
        state_dim_multiplier = 1




