import numpy as np


class Memory:
    def __init__(self, size, state_shape):
        self.counter = 0
        self.size = size
        self.state_shape = state_shape
        self.states = np.zeros((self.size, *self.state_shape), dtype=np.float32)
        self.new_states = np.zeros((self.size, *self.state_shape), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.uint8)

    def store_step(self, state, action, reward, new_state, done):
        """
        we store our states, actions, rewards, next states, terminal values
        :param state: state
        :param action: action
        :param reward: reward
        :param new_state: next state
        :param done: if this is the final transition
        :return: -
        """
        index = self.counter % self.size  # index at memory size
        self.states[index] = state
        self.new_states[index] = new_state
        self.actions[index] = action
        self.dones[index] = done
        self.rewards[index] = reward
        self.counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        batch_index = np.arange(batch_size, dtype=np.int32)
        # a batch observation? DUNNO what batch is (just a random observation)
        state_batch = self.states[batch]
        new_state_batch = self.new_states[batch]
        reward_batch = self.rewards[batch]
        done_batch = self.dones[batch]
        action_batch = self.actions[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch
