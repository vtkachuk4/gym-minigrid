import gym
import gym_minigrid
import os
import numpy as np
import matplotlib.pyplot as plt
from plot import plot_progress
from agent import EGreedyQLearner
from agent import UCBHQLearner
from agent import UCBHQLearnerOI
from helpers import init_optimal_q_table


def main():
    grid_list = ['MiniGrid-Empty-RandStartState-1x3-v0']
    # grid_list = ['MiniGrid-Empty-1x3-v0']

    num_episodes = 500
    horizon = 3
    c = 0.1
    p = 0.05

    avg_over = 10
    ep_chunk = 10
    plot_regret = True
    show_plot = True
    save_fig = True
    data_dir = f'data'
    x_axis = np.linspace(1, num_episodes, int(num_episodes / ep_chunk))
    # algorithms = [EGreedyQLearner, UCBHQLearner, UCBHQLearnerOI]
    algorithms = [UCBHQLearner, UCBHQLearnerOI]
    # algorithms = [UCBHQLearner]


    for grid in grid_list:
        main_dir = f'{data_dir}/{grid}_eps_{num_episodes}_avg' \
                   f'_{avg_over}'
        try:
            os.makedirs(main_dir)
        except:
            pass

        env = gym.make(grid)
        if plot_regret:
            optimal_q_table = init_optimal_q_table(env, horizon)

        for algorithm in algorithms:
            if plot_regret:
                avg_per_episode_regret = np.zeros(num_episodes // ep_chunk)
                optimal_rewards = np.zeros(num_episodes)
            else:
                avg_rewards_ep_chunk = np.zeros(num_episodes // ep_chunk)
            for avg_num in range(1, avg_over + 1):
                learner = algorithm(env, horizon)
                print(f'{grid}, algorithm: {learner.name}, '
                      f'run: {avg_num}')
                rewards, start_states = learner.learn(
                    num_episodes, ep_chunk=ep_chunk, c=c, p=p)

                if plot_regret:
                    for i, start_state in enumerate(start_states):
                        optimal_rewards[i] = optimal_q_table[1].loc[start_state].max()
                    per_episode_regret = optimal_rewards - rewards
                    per_episode_regret_smoothed = np.mean(per_episode_regret.reshape(
                        -1, ep_chunk), axis=1)
                    avg_per_episode_regret += per_episode_regret_smoothed / avg_over
                else:
                    rewards_ep_chunk = np.mean(rewards.reshape(-1, ep_chunk),
                                               axis=1)
                    avg_rewards_ep_chunk += rewards_ep_chunk / avg_over

            label = f'Algorithm: {learner.name}'
            if plot_regret:
                plot_progress(x_axis, avg_per_episode_regret, label)
                plt.ylabel('Per Episode Regret')
            else:
                plot_progress(x_axis, avg_rewards_ep_chunk, label)
                plt.ylabel('Per Episode Reward')
            plt.title(f'{grid}, Avg over {avg_over} runs, K: {num_episodes}, '
                      f'H: {horizon}, c: {c}, p: {p}')

        if save_fig:
            if plot_regret:
                save_loc_step = f'{main_dir}/per_episode_regret_plot.jpg'
            else:
                save_loc_step = f'{main_dir}/per_episode_reward_plot.jpg'
            plt.savefig(save_loc_step)
        if show_plot:
            plt.show()
        else:
            plt.close()
        optimal_policy = learner.get_optimal_policy()
        learner.pretty_print_policy(optimal_policy, 1)
        # e_greedy_q_learner.visualize_agent(num_episodes=1)
        env.close()


if __name__ == '__main__':
    main()
