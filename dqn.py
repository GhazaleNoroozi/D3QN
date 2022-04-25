import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch as T


class D3QN(nn.Module):

    def __init__(self, lr, state_dims, checkpoint_file, n_actions, V_dims=1, fc1_dims=512):
        super(D3QN, self).__init__()
        self.checkpoint_file = checkpoint_file
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.V_dims = V_dims

        # layer to handle the input of observation
        self.fc1 = nn.Linear(*self.state_dims, self.fc1_dims)

        # for dueling we need a value and advantage stream
        # value network tells agent what is the value of its current state
        # the advantage tells the relative advantage of each action in the state
        # each focus on different parts of the picture
        self.V = nn.Linear(self.fc1_dims, self.V_dims)   # 1 because our states are 1 dimension
        self.A = nn.Linear(self.fc1_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Predicts the state value and its actions advantages
        :param state: the state
        :return: the state value and its actions advantages
        """
        x = F.relu(self.fc1(state))
        x_V = self.V(x)
        x_A = self.A(x)
        return x_V, x_A


class D2QN(nn.Module):
    def __init__(self, lr, state_dims, n_actions, fc2_dims=256, fc1_dims=256):
        """
        Initializing the parameters
        :param lr:
        :param state_dims: shape of a state (shape of the input layer)
        :param fc1_dims: number of neurons in the first middle layer
        :param fc2_dims: number of neurons in the first middle layer
        :param n_actions: number of actions (shape of output layer)
        """
        super(D2QN, self).__init__()
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(*self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Predicts the state-action values for a state
        :param state: the state
        :return: predicted state-action values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class DQN(nn.Module):
    def __init__(self, lr, state_dims, fc1_dims, fc2_dims, n_actions):
        """
        Initializing the parameters
        :param lr:
        :param state_dims: shape of a state (shape of the input layer)
        :param fc1_dims: number of neurons in the first middle layer
        :param fc2_dims: number of neurons in the first middle layer
        :param n_actions: number of actions (shape of output layer)
        """
        super().__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.state_dims = state_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Predicts the state-action values for a state
        :param state: the state
        :return: predicted state-action values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
