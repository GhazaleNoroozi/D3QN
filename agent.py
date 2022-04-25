import torch as T
from memory import Memory
from dqn import DQN, D2QN, D3QN
import numpy as np
import os


class DeepAgent:
    """
    The superclass of all deep agents
    """
    def __init__(self, gamma, epsilon, lr, state_dims, batch_size, n_actions, mem_size, eps_end, eps_dec):
        """
        Initializing the class parameters (hyperparameters and more)
        :param gamma: discount factor
        :param epsilon: starting epsilon
        :param lr: learning rate
        :param state_dims: state shape to input the network
        :param batch_size: size of the batch
        :param n_actions: number of actions
        :param mem_size: size of the replay memory
        :param eps_end: min epsilon that we don't wanna get lower than that
        :param eps_dec: epsilon decay
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.n_actions = n_actions
        self.state_dims = state_dims
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.memory = Memory(self.mem_size, self.state_dims)
        # the first network
        self.Q_eval = DQN(self.lr, n_actions=self.n_actions, input_dims=self.state_dims, fc1_dims=256, fc2_dims=256)
        self.learn_step_counter = 0

    def choose_action(self, state):
        """
        Epsilon-greedy action-selection
        :param state: one state
        :return: the chosen action
        """
        if np.random.random() > self.epsilon:
            tensor_state = T.tensor([state], dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(tensor_state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_step(self, state, action, reward, new_state, done):
        self.memory.store_step(state, action, reward, new_state, done)

    def calculate_loss(self, states, actions, rewards, new_states, dones, indices):
        """
        Estimating the current state's and the next state's action values
        and calculating the mean squared error between the target and current value
        :param states: batch states
        :param actions: batch actions
        :param rewards: batch rewards
        :param new_states: batch states
        :param dones: batch terminals
        :param indices: batch indices
        :return: loss of the network
        """
        # we get our state-actions values for this state and next state to calculate the loss
        q_eval = self.Q_eval.forward(states)[indices, actions]
        q_next = self.Q_eval.forward(new_states)
        q_next[dones] = 0
        # maximum of the predicted action-values of the next state
        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]
        # the mean square error of the target value and the current value would be the loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        return loss

    def learn(self):
        # we don't want it to learn if our memory is still not full (in the first 'mem_size' steps)
        if self.memory.counter < self.batch_size:
            return

        # zero the gradiant on our optimizer
        self.Q_eval.optimizer.zero_grad()

        if isinstance(self, D2Agent) or isinstance(self, D3Agent):
            self.replace_target_network()

        # one random experience from the replay buffer called batch
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states).to(self.Q_eval.device)
        actions = T.tensor(actions).to(self.Q_eval.device)
        dones = T.tensor(dones).to(self.Q_eval.device)
        rewards = T.tensor(rewards).to(self.Q_eval.device)
        new_states = T.tensor(new_states).to(self.Q_eval.device)
        indices = np.arange(self.batch_size)

        # calculate loss
        loss = self.calculate_loss(states, actions, rewards, new_states, dones, indices)
        loss.backward()

        # update weights of the network
        self.Q_eval.optimizer.step()

        self.learn_step_counter += 1

        # decay epsilon
        self.epsilon -= self.eps_dec
        if self.epsilon < self.eps_min:
            self.epsilon = self.eps_min


class D2Agent(DeepAgent):
    """
    The agent for double deep q-learning
    """
    def __init__(self, gamma, epsilon, lr, n_actions, state_dims, batch_size,
                 eps_end=0.01, eps_dec=5e-4, mem_size=100000, replace=1000):

        super().__init__(gamma, epsilon, lr, state_dims, batch_size, n_actions, mem_size, eps_end, eps_dec)

        # first and second networks
        self.Q_eval = D2QN(self.lr, self.state_dims, self.n_actions)
        self.Q_next = D2QN(self.lr, self.state_dims, self.n_actions)
        self.replace_target_count = replace

    def replace_target_network(self):
        # every 'replace_target_count' steps, the second network gets updated
        if self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def calculate_loss(self, states, actions, rewards, new_states, dones, indices):
        # next state's action values using the first network
        # used for choosing the best action in the next state
        pred_actions_eval = self.Q_eval.forward(new_states)
        max_actions = T.argmax(pred_actions_eval, dim=1)

        # next state action values using the second network
        # the value of the best action which was calculated using the first network
        # will be achieved from the second network
        # used as the target value a.k.a the first term in the loss function
        new_pred_actions = self.Q_next.forward(new_states)
        q_next = new_pred_actions
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        # current state action values using the first network
        # used as the second term in the loss function
        pred_actions = self.Q_eval.forward(states)
        q_pred = pred_actions[indices, actions]

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        return loss


class D3Agent(DeepAgent):
    def __init__(self, gamma, epsilon, lr, n_actions, state_dims, batch_size, eps_end=0.01,
                 eps_dec=5e-4, mem_size=100000, replace=1000):

        super().__init__(gamma, epsilon, lr, state_dims, batch_size, n_actions, mem_size, eps_end, eps_dec)

        self.replace_target_count = replace
        self.Q_eval = D3QN(self.lr, self.state_dims, os.path.join('d3qn', 'first'), self.n_actions)
        self.Q_next = D3QN(self.lr, self.state_dims, os.path.join('d3qn', 'second'), self.n_actions)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def calculate_loss(self, states, actions, rewards, new_states, dones, indices):
        # Duelling Double Deep Q-Learning. Using the state value and action advantage
        # instead of a single state-action value shows the presence of duelling Q-learning.
        # the two networks, one used for calculating the best action,
        # and the second one used for getting the value of that action to avoid overestimation
        # shows the presence of double Q-learning

        # finding the best action
        Vs_eval, As_eval = self.Q_eval.forward(new_states)
        q_eval = T.add(Vs_eval, (As_eval - As_eval.mean(dim=1, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=1)

        # finding the value of the best action
        new_Vs, new_As = self.Q_next.forward(new_states)
        q_next = T.add(new_Vs, (new_As - new_As.mean(dim=1, keepdim=True)))
        q_next[dones] = 0.0
        # calculating the target term, the first term in the loss function
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        # calculating the current value, the second term in the loss function
        Vs, As = self.Q_eval.forward(states)
        q_pred = T.add(Vs, (As - As.mean(dim=1, keepdim=True)))[indices, actions]
        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        return loss

    def choose_action(self, state):
        """
        Epsilon-greedy action-selection
        :param state: one state
        :return: the chosen action
        """
        if np.random.random() > self.epsilon:
            state = state[np.newaxis, :]
            tensor_state = T.tensor(state).to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(tensor_state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action
