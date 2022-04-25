from agent import DeepAgent, D2Agent, D3Agent

agent1 = DeepAgent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=4, state_dims=[8], batch_size=64,
                  eps_end=0.01, eps_dec=5e-4, mem_size=100000)
file_name1 = 'LunarLander1'
agent2 = D2Agent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=4, state_dims=[8], batch_size=64,
                eps_end=0.01, eps_dec=0.001, mem_size=1000000, replace=100)
file_name2 = 'LunarLander2'
agent3 = D3Agent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=4, state_dims=[8], batch_size=64,
                eps_end=0.01, eps_dec=0.001, mem_size=1000000, replace=100)
file_name3 = 'LunarLander3'

AGENT = agent1
# FILENAME = file_name1
FILENAME = 'lr_dqn'
EPISODES = 500
IS_RENDER = False

