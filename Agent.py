import torch as T
from Memory import Memory
from DeepQLearning import DQN, D2QN, D3QN
import numpy as np
import os


class DeepAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, mem_size, eps_end, eps_dec):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.memory = Memory(self.mem_size, input_dims)

        self.Q_eval = DQN(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=256, fc2_dims=256)
        self.learn_step_counter = 0

    def choose_action(self, observation):
        """
        Epsilon-greedy action-selection
        :param observation: one state, action, next_state, reward, done
        :return: the chosen action
        """
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_step(state, action, reward, new_state, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def calculate_loss(self, states, actions, rewards, new_states, dones, indices):
        # we get our state-actions values for this state and next state to calculate the loss
        q_eval = self.Q_eval.forward(states)[indices, actions]
        q_next = self.Q_eval.forward(new_states)
        q_next[dones] = 0
        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        return loss

    def learn(self):
        # we don't want it to learn if our memory is still not full so the first mem_size steps
        if self.memory.counter < self.batch_size:
            return

        # zero the gradiant on our optimizer
        self.Q_eval.optimizer.zero_grad()
        # find the current memory size (some at the end of the memory might still be zero a.k.a. empty)
        if isinstance(self, D2Agent) or isinstance(self, D3Agent):
            # print('do be agent 2 or 3 for real working')
            self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.Q_eval.device)
        actions = T.tensor(action).to(self.Q_eval.device)
        dones = T.tensor(done).to(self.Q_eval.device)
        rewards = T.tensor(reward).to(self.Q_eval.device)
        new_states = T.tensor(new_state).to(self.Q_eval.device)
        indices = np.arange(self.batch_size)

        loss = self.calculate_loss(states, actions, rewards, new_states, dones, indices)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class D2Agent(DeepAgent):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, batch_size,
                 eps_end=0.01, eps_dec=5e-4, mem_size=100000, replace=1000):

        super().__init__(gamma, epsilon, lr, input_dims, batch_size, n_actions, mem_size, eps_end, eps_dec)

        self.Q_eval = D2QN(self.lr, self.input_dims, self.n_actions)
        self.Q_next = D2QN(self.lr, self.input_dims, self.n_actions)
        self.replace_target_count = replace

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def calculate_loss(self, states, actions, rewards, new_states, dones, indices):
        pred_actions = self.Q_eval.forward(states)
        new_pred_actions = self.Q_next.forward(new_states)

        # Double Deep Q learning
        pred_actions_eval = self.Q_eval.forward(new_states)

        q_pred = pred_actions[indices, actions]
        q_next = new_pred_actions  # for all actions
        q_eval = pred_actions_eval
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        return loss


class D3Agent(DeepAgent):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, batch_size, eps_end=0.01,
                 eps_dec=5e-4, mem_size=100000, replace=1000):

        super().__init__(gamma, epsilon, lr, input_dims, batch_size, n_actions, mem_size, eps_end, eps_dec)

        self.replace_target_count = replace
        self.Q_eval = D3QN(self.lr, self.input_dims, os.path.join('d3qn', 'first'), self.n_actions)
        self.Q_next = D3QN(self.lr, self.input_dims, os.path.join('d3qn', 'second'), self.n_actions)

    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()

    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def calculate_loss(self, states, actions, rewards, new_states, dones, indices):
        V_s, A_s = self.Q_eval.forward(states)
        new_V_s, new_A_s = self.Q_next.forward(new_states)

        #   Double Deep Q learning
        V_s_eval, A_s_eval = self.Q_eval.forward(new_states)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(new_V_s, (new_A_s - new_A_s.mean(dim=1, keepdim=True)))  # for all actions
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        return loss

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis, :]
            state = T.tensor(observation).to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action
