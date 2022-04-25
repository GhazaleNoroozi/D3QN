import gym
import numpy as np
import matplotlib.pyplot as plt
import winsound
from garbage import Plot
from Agent import DeepAgent, D2Agent, D3Agent


def main(agent, file_name):
    env = gym.make('LunarLander-v2')

    if isinstance(agent, D3Agent):
        load_checkpoint = False
        if load_checkpoint:
            agent.load_models()

    scores, avg_scores, eps_history = [], [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state, action, reward, new_state, int(done))
            agent.learn()
            # env.render()
            state = new_state

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print('episode ', i, 'score %.2f' % score, 'avg score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

        if isinstance(agent, D3Agent) and i > 10 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)

    return scores, avg_scores

    # x = [i + 1 for i in range(n_games)]
    # Plot.plotLearning(x, scores, eps_history, file_name)


if __name__ == '__main__':
    agent1 = DeepAgent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=4, input_dims=[8], batch_size=64,
                      eps_end=0.01, eps_dec=5e-4, mem_size=100000)
    agent2 = D2Agent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=4, input_dims=[8], batch_size=64,
                    eps_end=0.01, eps_dec=0.001, mem_size=1000000, replace=100)
    agent3 = D3Agent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=4, input_dims=[8], batch_size=64,
                    eps_end=0.01, eps_dec=0.001, mem_size=1000000, replace=100)
    file_name1 = 'LunarLander1'
    file_name2 = 'LunarLander2'
    file_name3 = 'LunarLander3'

    scores, avg = main(agent3, file_name3)
    winsound.Beep(2500, 1000)
    plt.plot(avg)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Sum of Rewards")
    plt.legend(loc='best')
    plt.savefig('diagrams/' + file_name1 + '_avg')
    plt.clf()
    plt.plot(scores)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Sum of Rewards")
    plt.legend(loc='best')
    plt.savefig('diagrams/' + file_name1)

    # for gamma in [0.8, 0.9, 0.99, 0.999]:
    #     agent1 = DeepAgent(gamma=gamma, epsilon=1.0, lr=0.001, n_actions=4, input_dims=[8], batch_size=64,
    #                    eps_end=0.01, eps_dec=5e-4, mem_size=100000)
    #     scores, avg_scores = main(agent1, file_name1)
    #     winsound.Beep(2500, 1000)
    #     plt.plot(scores, label=f'gamma={gamma}')
    #     # plt.savefig('diagrams/' + file_name)
    # plt.xlabel("Number of Episodes")
    # plt.ylabel("Sum of Rewards")
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig('diagrams/' + file_name + '_avg')
    # plt.clf()
