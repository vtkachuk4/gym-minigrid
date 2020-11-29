import gym
import gym_minigrid
import os
import numpy as np
import matplotlib.pyplot as plt
from plot import plot_progress
from agent import EGreedyQLearner


def main():
    grid_list = ['MiniGrid-Empty-1x10-v0']

    num_episodes = 20
    learning_rate = 0.1
    discount_rate = 0.9
    exploration_rate = 0.1
    horizon = 3

    avg_over = 1
    ep_chunk = 2
    show_plot = False
    save_fig = True
    data_dir = f'data'
    x_axis = np.linspace(1, num_episodes, int(num_episodes / ep_chunk))

    for grid in grid_list:
        main_dir = f'{data_dir}/{grid}_eps_{num_episodes}_avg' \
                   f'_{avg_over}'
        try:
            os.makedirs(main_dir)
        except:
            pass

        env = gym.make(grid)
        e_greedy_q_learner = EGreedyQLearner(env)

        avg_steps_ep_chunk = np.zeros(num_episodes // ep_chunk)
        avg_rewards_ep_chunk = np.zeros(num_episodes // ep_chunk)
        for avg_num in range(1, avg_over + 1):
            print(f'{grid}, run: {avg_num}')
            steps, rewards = e_greedy_q_learner.q_learning(
                num_episodes, learning_rate, discount_rate, exploration_rate,
                horizon, ep_chunk)

            rewards_ep_chunk = np.mean(rewards.reshape(-1, ep_chunk),
                                           axis=1)
            steps_ep_chunk = np.mean(steps.reshape(-1, ep_chunk), axis=1)
            avg_steps_ep_chunk += steps_ep_chunk / avg_over
            avg_rewards_ep_chunk += rewards_ep_chunk / avg_over

        label = f'Algorithm: e-greedy Q-learning)'
        # plot_progress(x_axis, avg_steps_ep_chunk, label)
        # plt.title(f'Steps Per Episode, Avg over {avg_over} runs, {grid}')
        plot_progress(x_axis, avg_rewards_ep_chunk, label)
        plt.title(f'Return Per Episode, Avg over {avg_over} runs, {grid}')

        if save_fig:
            save_loc_step = f'{main_dir}/reward_progress_plot.jpg'
            plt.savefig(save_loc_step)
        if show_plot:
            plt.show()
        else:
            plt.close()
        optimal_policy = e_greedy_q_learner.get_optimal_policy()
        e_greedy_q_learner.pretty_print_policy(optimal_policy)
        e_greedy_q_learner.visualize_agent()
        env.close()


if __name__ == '__main__':
    main()
