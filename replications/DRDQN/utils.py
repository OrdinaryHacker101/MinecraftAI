import numpy as np

class ReplayBuffer():
    def __init__(self, mem_size, img_shape):
        
        self.mem_cntr = 0
        self.mem_size = mem_size
        self.img_shape = img_shape

        self.state_mem = np.zeros((self.mem_size, *img_shape),
                                  dtype=np.float32)
        self.n_state_mem = np.zeros((self.mem_size, *img_shape),
                                    dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size,
                                   dtype=np.float32)
        self.reward_mem = np.zeros(self.mem_size,
                                   dtype=np.float32)
        self.done_mem = np.zeros(self.mem_size,
                                 dtype=np.bool)

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
    
