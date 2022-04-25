import gym
import numpy as np
import matplotlib.pyplot as plt
import winsound
import constants
from agent import DeepAgent


def main(agent):
    # creating the lunar lander environment from gym
    env = gym.make('LunarLander-v2')

    # if isinstance(agent, D3Agent):
    #     load_checkpoint = False
    #     if load_checkpoint:
    #         agent.load_models()

    summed_rewards, avg_summed_rewards = [], []
    episodes = constants.EPISODES
    for i in range(episodes):
        summed_reward = 0
        done = False
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            summed_reward += reward
            agent.store_step(state, action, reward, new_state, int(done))
            agent.learn()
            if constants.IS_RENDER:
                env.render()
            state = new_state

        summed_rewards.append(summed_reward)
        avg_summed_reward = np.mean(summed_rewards[-100:])
        avg_summed_rewards.append(avg_summed_reward)
        print(f'episode: {i} cumulative reward: {summed_reward} average last 100: {avg_summed_reward}')

        # if isinstance(agent, D3Agent) and i > 10 and i % 10 == 0:
        #     agent.save_models()

    return summed_rewards, avg_summed_rewards


def plot_and_save(reward_list, filename):
    plt.plot(reward_list)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Sum of Rewards")
    plt.legend(loc='best')
    plt.savefig('diagrams/' + filename)
    plt.clf()


if __name__ == '__main__':
    agent = constants.AGENT
    # rewards, avg_rewards = main(agent)
    # winsound.Beep(2500, 1000)
    file_name = constants.FILENAME
    # plot_and_save(avg_rewards, file_name + '_avg')
    # plot_and_save(rewards, file_name)

    for gamma in [0.8, 0.9, 0.99, 0.999]:
        agent = DeepAgent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=4, state_dims=[8], batch_size=64,
                  eps_end=0.01, eps_dec=5e-4, mem_size=100000)
        rewards, avg_rewards = main(agent)
        winsound.Beep(2500, 500)
        plt.plot(rewards, label=f'gamma={gamma}')
        plt.legend(loc='best')

    plt.xlabel("Number of Episodes")
    plt.ylabel("Sum of Rewards")
    plt.savefig('diagrams/' + file_name)
    plt.clf()
