import gym
import gym_minigrid
import os
import numpy as np
import matplotlib.pyplot as plt
from plot import plot_progress
from agent import UCBHQLearner, UCBHQLearnerOI, UCBHQLearnerOIwoA1
from helpers import init_optimal_q_table


def main():
    grid_list = ['MiniGrid-Empty-RandStartState-1x3-v0']

    num_episodes = 300
    horizon = 3
    c = 0.1
    p = 0.05

    ep_chunk = 5
    avg_over = 50
    show_plot = True
    algorithms = [UCBHQLearner, UCBHQLearnerOIwoA1, UCBHQLearnerOI]
    plot_line_format = ['-', '--', '-.']

    for grid in grid_list:
        main_dir = f'data/{grid}_K_{num_episodes}_H_{horizon}_c_{c}_p_' \
                   f'{p}_avg_{avg_over}'
        try:
            os.makedirs(main_dir)
        except:
            pass

        env = gym.make(grid)
        optimal_q_table = init_optimal_q_table(env, horizon)

        for idx, algorithm in enumerate(algorithms):
            optimal_rewards = np.zeros(num_episodes)
            all_regret = []
            for avg_num in range(1, avg_over + 1):
                learner = algorithm(env, horizon)
                print(f'{grid}, algorithm: {learner.name}, '
                      f'run: {avg_num}')
                rewards, start_states = learner.learn(
                    num_episodes=num_episodes, ep_chunk=ep_chunk, c=c, p=p)

                for i, start_state in enumerate(start_states):
                    optimal_rewards[i] = optimal_q_table[1].loc[
                        start_state].max()
                per_episode_regret = optimal_rewards - rewards
                per_episode_regret_smoothed = np.mean(
                    per_episode_regret.reshape(-1, ep_chunk), axis=1)
                all_regret.append(per_episode_regret_smoothed)

            label = f'{learner.name}'
            plot_progress(all_regret, ep_chunk, plot_line_format[idx], label)
            np.save(f'{main_dir}/per_episode_regret_data', all_regret)
        plt.ylabel('Per Episode Regret')
        plt.title(f'Per Episode Regret Plot')
        save_loc_step = f'{main_dir}/per_episode_regret_plot.jpg'
        plt.savefig(save_loc_step)
        plt.show() if show_plot else plt.close()
        learner.visualize_agent(num_episodes=1)
        env.close()


if __name__ == '__main__':
    main()
