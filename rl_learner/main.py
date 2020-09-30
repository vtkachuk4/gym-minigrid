import numpy as np
import pandas as pd
import gym
import random
import time
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gym_minigrid


def test_cur_policy(q_table, test_eps=1):
    avg_reward_all_episodes = 0
    for eps in range(test_eps):
        state = env.reset()

        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            action = q_table.columns[q_table.loc[state].argmax()]

            new_state, reward, done, info = env.step(action)
            state = new_state
            rewards_current_episode += reward

            if done == True:
                break

        avg_reward_all_episodes += rewards_current_episode / test_eps
    return avg_reward_all_episodes


def teach(q_table, policy, theta=0.1):
    for i in range(1, policy.index.size + 1):
        for j in range(1, policy.columns.size + 1):
            state = (j, i)
            target_action = policy.loc[i, j]
            if target_action != 'NaN':
                q_table.loc[state, target_action] = q_table.loc[
                                                        state].max() + theta
    return q_table


def get_optimal_policy(q_table, terminal_states=None):
    height = q_table[0].loc[:, 1].size
    width = q_table[0].loc[1, :].size
    optimal_policy = pd.DataFrame(np.zeros((width, height)),
                                  index=np.linspace(1, height, height,
                                                    dtype=int),
                                  columns=np.linspace(1, width, width,
                                                      dtype=int))
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            # remembering indices for q_table are x, y values
            optimal_policy.loc[i, j] = \
                q_table.columns[q_table.loc[j, i].argmax()]
    if terminal_states:
        for terminal_state in terminal_states:
            optimal_policy.loc[terminal_state[1], terminal_state[0]] = 'NaN'
    return optimal_policy


def pretty_print_policy(policy):
    arrow_dict = {0: '>', 1: 'v', 2: '<', 3: '^', 'NaN': 0}
    height = policy.index.size
    width = policy.columns.size
    pretty_policy = pd.DataFrame(np.zeros((width, height)),
                                 index=np.linspace(1, height, height,
                                                   dtype=int),
                                 columns=np.linspace(1, width, width,
                                                     dtype=int))
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            pretty_policy.loc[i, j] = arrow_dict[policy.loc[i, j]]
    print(pretty_policy)


def check_if_optimal_policy(policy):
    height = policy.index.size
    width = policy.columns.size
    return_val = True
    # check edges
    for i in range(1, height):
        if policy.loc[i, width] != 1:
            return_val = False
    for i in range(1, width):
        if policy.loc[height, i] != 0:
            return_val = False

    # check inner grid cells to be right or down
    for i in range(1, height):
        for j in range(1, width):
            if policy.loc[i, j] != 0 and policy.loc[i, j] != 1:
                return_val = False

    return return_val


# Q-learning algorithm
def q_learning(q_table, num_episodes, max_steps_per_episode, learning_rate,
               discount_rate, test_policy=True, print_rew=False):
    rewards_all_episodes = np.zeros(num_episodes)
    test_reward_per_ep_chunk = np.zeros(num_episodes // ep_chunk)
    steps_per_episode = np.zeros(num_episodes)

    for episode in range(1, num_episodes + 1):
        # print(f'we are on episode: {episode}')
        if test_policy:
            if episode % ep_chunk == 0:
                test_reward_per_ep_chunk[episode // ep_chunk - 1] = \
                    test_cur_policy(q_table)

        state = env.reset()

        rewards_current_episode = 0
        step = 1

        while True:
            # env.render()

            # Exploration and Exploitation
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = q_table.columns[q_table.loc[state].argmax()]
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            # update Q-table
            q_table.loc[state, action] = q_table.loc[state, action] * (
                    1 - learning_rate) + learning_rate * (
                                                 reward + discount_rate *
                                                 q_table.loc[new_state].max())

            state = new_state
            rewards_current_episode += reward

            if done == True or step == max_steps_per_episode:
                steps_per_episode[episode - 1] = step
                break

            step += 1

        rewards_all_episodes[episode - 1] = rewards_current_episode
        if print_rew:
            if episode % ep_chunk == 0 and episode != 0:
                avg_rew = sum(
                    rewards_all_episodes[episode - ep_chunk:episode]) / \
                          ep_chunk
                print(f'Episode: {episode}, avg reward for last {ep_chunk} '
                      f'episodes: {avg_rew}')

    # reward_per_ep_chunk = np.split(np.array(rewards_all_episodes),
    #                                num_episodes / ep_chunk)
    reward_per_ep_chunk = np.mean(rewards_all_episodes.reshape(-1,
                                                                ep_chunk), axis=1)
    # reward_per_ep_chunk = [sum(r / ep_chunk) for r in
    #                        reward_per_ep_chunk]

    # steps_per_ep_chunk = np.split(np.array(steps_per_episode),
    #                               num_episodes / ep_chunk)
    steps_per_ep_chunk = np.mean(steps_per_episode.reshape(-1,
                                                                   ep_chunk),
                               axis=1)
    # steps_per_ep_chunk = [sum(step / ep_chunk) for step in
    #                       steps_per_ep_chunk]

    return steps_per_ep_chunk, reward_per_ep_chunk, test_reward_per_ep_chunk


def plot_step_progress(x_axis, save_loc, title, *argv):
    alpha = 0.5
    linewidth = 2
    iters = len(argv) // 2

    plt.rcParams["figure.figsize"] = (10, 5)
    for i in range(iters):
        plt.plot(x_axis, argv[i * 2 + 0], label=argv[i * 2 + 1], alpha=alpha,
                 linewidth=linewidth)

    plt.xlabel('Episode number')
    plt.ylabel('Steps taken')
    plt.title(title)
    plt.subplots_adjust(bottom=0.3)
    plt.legend(bbox_to_anchor=(0, -0.1, 1, -0.1), loc="upper left",
               mode="expand", borderaxespad=0, ncol=2)
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    if save_loc:
        plt.savefig(save_loc)


def plot_reward_progress(x_axis, y_axis_1, y_axis_2=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

    axes[0].plot(x_axis, y_axis_1)
    axes[0].set(xlabel='episodes', ylabel='Avg Total Reward')
    if y_axis_2 is not None:
        axes[1].plot(x_axis, y_axis_2)
        axes[1].set(xlabel='episodes', ylabel='Avg Total Reward')
    fig.tight_layout()
    plt.show()


def plot_all_reward_progress(x_axis, save_loc, title=None, *argv):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    alpha = 0.5
    linewidth = 2
    iters = len(argv) // 3

    for i in range(iters):
        axes[0].plot(x_axis, argv[i * 3], label=argv[i * 3 + 2], alpha=alpha,
                     linewidth=linewidth)
        axes[1].plot(x_axis, argv[i * 3 + 1], label=argv[i * 3 + 2],
                     alpha=alpha,
                     linewidth=linewidth)

    axes[0].set(xlabel='episodes', ylabel='Avg Total Reward')
    axes[1].set(xlabel='episodes', ylabel='Avg Total Reward')
    if title:
        axes[0].set(title='e-Greedy ' + title)
        axes[1].set(title='Greedy ' + title)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

    if save_loc:
        plt.savefig(save_loc)


def plot_reward_progress_overlay(x_axis, y_axis_1_1, y_axis_2_1, label_1,
                                 y_axis_1_2=None,
                                 y_axis_2_2=None, label_2=None, y_axis_1_3=None,
                                 y_axis_2_3=None, label_3=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    alpha = 0.5
    linewidth = 2
    axes[0].plot(x_axis, y_axis_1_1, label=label_1, alpha=alpha,
                 linewidth=linewidth)
    if y_axis_1_2:
        axes[0].plot(x_axis, y_axis_1_2, label=label_2, alpha=alpha,
                     linewidth=linewidth)
    if y_axis_1_3:
        axes[0].plot(x_axis, y_axis_1_3, label=label_3, alpha=alpha,
                     linewidth=linewidth)
    axes[0].set(xlabel='episodes', ylabel='Avg Total Reward')
    axes[1].plot(x_axis, y_axis_2_1, label=label_1, alpha=alpha,
                 linewidth=linewidth)
    if y_axis_2_2:
        axes[1].plot(x_axis, y_axis_2_2, label=label_2, alpha=alpha,
                     linewidth=linewidth)
    if y_axis_2_3:
        axes[1].plot(x_axis, y_axis_2_3, label=label_3, alpha=alpha,
                     linewidth=linewidth)
    axes[1].set(xlabel='episodes', ylabel='Avg Total Reward')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def init_q_table(height, width, num_actions, random=True, min=0, max=0,
                 terminal_states=None):
    states = []
    for a in range(1, height + 1):
        for b in range(1, width + 1):
            states.append((a, b))
    val_range = max - min
    if random:
        assert max > min, 'max must be larger than min'
        init_vals = np.random.rand(len(states), num_actions) * val_range + min
    else:
        assert max == min, 'max must equal min for non-random init'
        init_vals = np.zeros((len(states), num_actions)) + min

    q_table = pd.DataFrame(init_vals,
                           index=pd.MultiIndex.from_tuples(states),
                           columns=np.linspace(
                               0, num_actions - 1, num_actions, dtype=int))
    if terminal_states:
        for terminal_state in terminal_states:
            q_table.loc[terminal_state] = 0
    return q_table


def visualize_agent(q_table, num_episodes=2, max_steps_per_episode=100):
    for episode in range(num_episodes):
        state = env.reset()

        for step in range(max_steps_per_episode):
            env.render()

            action = q_table.columns[q_table.loc[state].argmax()]
            new_state, reward, done, info = env.step(action)
            print(reward)

            if done:
                env.render()
                if reward == 1:
                    print("*****You've reached your goal!*****")
                    time.sleep(1)
                else:
                    print("*****You fell through a hole!*****")
                    time.sleep(1)
                clear_output(wait=True)
                break
            state = new_state


def gen_optimal_policy(height, width, goal_state):
    optimal_policy = pd.DataFrame(np.zeros((width, height)),
                                  index=np.linspace(1, height, height,
                                                    dtype=int),
                                  columns=np.linspace(1, width, width,
                                                      dtype=int))
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            if j < width:
                optimal_policy.loc[i, j] = 0
            else:
                optimal_policy.loc[i, j] = 1
    optimal_policy.loc[goal_state[1], goal_state[0]] = 'NaN'
    return optimal_policy


if __name__ == '__main__':
    grid = 'MiniGrid-Empty-8x8-v0'
    env = gym.make(grid)

    num_actions = env.action_space.n
    num_orientations = 4
    actual_grid_height = env.height - 2
    actual_grid_width = env.width - 2
    terminal_states = [(actual_grid_width, actual_grid_height)]
    goal_state = terminal_states[-1]

    # Testing stuff out
    q_table_init_z = init_q_table(actual_grid_height, actual_grid_width,
                                  num_actions, random=False)
    print(q_table_init_z)
    q_table_init_r = init_q_table(actual_grid_height, actual_grid_width,
                                  num_actions, random=True, min=-11, max=-10,
                                  terminal_states=terminal_states)
    print(q_table_init_r)
    policy_init = get_optimal_policy(q_table_init_r, terminal_states)
    print(policy_init)
    pretty_print_policy(policy_init)

    teaching_policy = gen_optimal_policy(actual_grid_height,
                                         actual_grid_width, goal_state)
    pretty_print_policy(teaching_policy)


    ep_chunk = 1
    max_steps_per_episode = 100

    learning_rate = 0.1
    discount_rate = 0.9
    exploration_rate = 0.1

    num_episodes = 5
    theta_list = [0.05, 0.15]
    x_axis = np.linspace(1, num_episodes, int(num_episodes / ep_chunk))
    plot_rew = False
    plot_step = True
    show_plot = False
    teaching = False
    no_teaching = True
    init_random = True
    avg_over = 1
    # q_init_list = [(100, 100), (10, 10), (0.5, 0.5), (0, 0), (-10, -10),
    #                (-100, -100)]
    q_init_list = [(100, 101), (10, 11), (0, 1), (-1, 0), (-11, -10),
                   (-101, -100)]
    grid_list = ['MiniGrid-Empty-Reward-0-1-5x5-v0', 'MiniGrid-Empty-Reward-0-1-10x10-v0',
                 'MiniGrid-Empty-Reward-0-1-20x20-v0']

    for grid in grid_list:
        main_dir = f'{grid}_avg_{avg_over}_random_{init_random}/'
        try:
            os.mkdir(main_dir)
        except:
            pass

        env = gym.make(grid)
        num_actions = env.action_space.n
        num_orientations = 4
        actual_grid_height = env.height - 2
        actual_grid_width = env.width - 2
        terminal_states = [(actual_grid_width, actual_grid_height)]
        goal_state = terminal_states[-1]
        for q_init in q_init_list:
            title = f'{grid}, Avg over {avg_over} runs, Random = {init_random}'
            if no_teaching:
                avg_steps_all_ep_nt = np.zeros(num_episodes // ep_chunk)
                for avg_num in range(1, avg_over + 1):
                    print(f'No Teaching {grid} Cons Init ({q_init[0]},'
                          f' {q_init[1]}) '
                          f'run: {avg_num}')
                    q_table_nt = init_q_table(actual_grid_height, actual_grid_width,
                                                 num_actions,
                                                 random=init_random,
                                                 min=q_init[0],
                                                 max=q_init[1],
                                                 terminal_states=terminal_states)
                    steps_all_ep_nt, rew_all_ep_nt, t_rew_all_ep_nt = \
                        q_learning(
                            q_table_nt,
                            num_episodes,
                            max_steps_per_episode,
                            learning_rate,
                            discount_rate)

                    avg_steps_all_ep_nt += steps_all_ep_nt / avg_over

                    # policy_nt = get_optimal_policy(q_table_nt,
                    #                                   terminal_states=terminal_states)
                    # pretty_print_policy(policy_nt)

            nt_label = f'No Teaching Init Range: ({q_init[0]}, {q_init[1]})'
            save_dir = main_dir
            try:
                os.mkdir(save_dir)
            except:
                pass
            if plot_step:
                save_loc_step = save_dir + (f'nt_step.jpg')
                plot_step_progress(x_axis, save_loc_step, title,
                                   avg_steps_all_ep_nt, nt_label)
        if show_plot:
            plt.show()
        else:
            plt.close()

    teaching = True
    no_teaching = False

    for grid in grid_list:
        main_dir = f'{grid}_avg_{avg_over}_random_{init_random}/'
        try:
            os.mkdir(main_dir)
        except:
            pass

        env = gym.make(grid)
        num_actions = env.action_space.n
        num_orientations = 4
        actual_grid_height = env.height - 2
        actual_grid_width = env.width - 2
        terminal_states = [(actual_grid_width, actual_grid_height)]
        goal_state = terminal_states[-1]
        for theta in theta_list:
            for q_init in q_init_list:
                title = f'{grid}, Avg over {avg_over} runs, Theta = {theta}, Random = {init_random}'
                if no_teaching:
                    avg_steps_all_ep_nt = np.zeros(num_episodes // ep_chunk)
                    for avg_num in range(1, avg_over + 1):
                        print(
                            f'No Teaching {grid} Cons Init ({q_init[0]}, {q_init[1]}) '
                            f'run: {avg_num}')
                        q_table_nt = init_q_table(actual_grid_height,
                                                  actual_grid_width,
                                                  num_actions, random=init_random,
                                                  min=q_init[0],
                                                  max=q_init[1],
                                                  terminal_states=terminal_states)
                        steps_all_ep_nt, rew_all_ep_nt, t_rew_all_ep_nt = \
                            q_learning(
                                q_table_nt,
                                num_episodes,
                                max_steps_per_episode,
                                learning_rate,
                                discount_rate)

                        avg_steps_all_ep_nt += steps_all_ep_nt / avg_over

                        # policy_nt = get_optimal_policy(q_table_nt,
                        #                                   terminal_states=terminal_states)
                        # pretty_print_policy(policy_nt)

                if teaching:
                    avg_steps_all_ep_t = np.zeros(num_episodes // ep_chunk)
                    for avg_num in range(1, avg_over + 1):
                        print(
                            f'Teaching {grid}, Theta = {theta}, Cons Init ({q_init[0]},'
                            f' {q_init[1]}) run:'
                            f' {avg_num}')
                        q_table_t = init_q_table(actual_grid_height,
                                                 actual_grid_width,
                                                 num_actions, random=init_random,
                                                 min=q_init[0],
                                                 max=q_init[1],
                                                 terminal_states=terminal_states)
                        target_policy = gen_optimal_policy(actual_grid_height,
                                                           actual_grid_width,
                                                           goal_state)
                        q_table_t = teach(q_table_t, target_policy, theta=theta)
                        steps_all_ep_t, rew_all_ep_t, t_rew_all_ep_t = \
                            q_learning(
                                q_table_t,
                                num_episodes,
                                max_steps_per_episode,
                                learning_rate,
                                discount_rate)

                        avg_steps_all_ep_t += steps_all_ep_t / avg_over

                        # policy_t = get_optimal_policy(q_table_t,
                        #                                terminal_states=terminal_states)
                        # pretty_print_policy(policy_t)

                nt_label = f'No Teaching Init Range: ({q_init[0]}, {q_init[1]})'
                t_label = f'Teaching Init Range: ({q_init[0]}, {q_init[1]})'
                save_dir = main_dir
                try:
                    os.mkdir(save_dir)
                except:
                    pass
                save_loc_rew = save_dir + (f'rew_theta_{theta}.jpg')
                if plot_rew:
                    plot_all_reward_progress(x_axis, save_loc_rew, title,
                                             rew_all_ep_nt,
                                             t_rew_all_ep_nt, nt_label,
                                             rew_all_ep_t, t_rew_all_ep_t,
                                             t_label)
                if plot_step:
                    if plot_step:
                        if teaching and no_teaching:
                            save_loc_step = save_dir + (f't_nt_step_theta'
                                                        f'_{theta}.jpg')
                            plot_step_progress(x_axis, save_loc_step, title,
                                               avg_steps_all_ep_nt, nt_label,
                                               avg_steps_all_ep_t, t_label)
                        elif teaching:
                            save_loc_step = save_dir + (f't_step_theta'
                                                        f'_{theta}.jpg')
                            plot_step_progress(x_axis, save_loc_step, title,
                                               avg_steps_all_ep_t, t_label)
                        else:
                            save_loc_step = save_dir + (f'nt_step.jpg')
                            plot_step_progress(x_axis, save_loc_step, title,
                                               avg_steps_all_ep_nt, nt_label)
            if show_plot:
                plt.show()
            else:
                plt.close()

        # Actually render environment and watch the agent
        # visualize_agent(q_table_t, num_episodes=2)


    theta_list = [0.005, 0.015]
    # q_init_list = [(10, 10), (0, 0), (-5, -5), (-10, -10),
    #                (-11, -11), (-100, -100)]
    q_init_list = [(10, 11), (0, 1), (-10, 0), (-10, -9),
                   (-12, -11), (-101, -100)]
    grid_list = ['MiniGrid-Empty-Reward--1-0-5x5-v0',
                 'MiniGrid-Empty-Reward--1-0-10x10-v0',
                 'MiniGrid-Empty-Reward--1-0-20x20-v0']

    for grid in grid_list:
        main_dir = f'{grid}_avg_{avg_over}_random_{init_random}/'
        try:
            os.mkdir(main_dir)
        except:
            pass

        env = gym.make(grid)
        num_actions = env.action_space.n
        num_orientations = 4
        actual_grid_height = env.height - 2
        actual_grid_width = env.width - 2
        terminal_states = [(actual_grid_width, actual_grid_height)]
        goal_state = terminal_states[-1]
        for q_init in q_init_list:
            title = f'{grid}, Avg over {avg_over} runs, Random = {init_random}'
            if no_teaching:
                avg_steps_all_ep_nt = np.zeros(num_episodes // ep_chunk)
                for avg_num in range(1, avg_over + 1):
                    print(f'No Teaching {grid} Cons Init ({q_init[0]}, {q_init[1]}) '
                          f'run: {avg_num}')
                    q_table_nt = init_q_table(actual_grid_height, actual_grid_width,
                                                 num_actions, random=init_random,
                                                 min=q_init[0],
                                                 max=q_init[1],
                                                 terminal_states=terminal_states)
                    steps_all_ep_nt, rew_all_ep_nt, t_rew_all_ep_nt = \
                        q_learning(
                            q_table_nt,
                            num_episodes,
                            max_steps_per_episode,
                            learning_rate,
                            discount_rate)

                    avg_steps_all_ep_nt += steps_all_ep_nt / avg_over

                    # policy_nt = get_optimal_policy(q_table_nt,
                    #                                   terminal_states=terminal_states)
                    # pretty_print_policy(policy_nt)

            nt_label = f'No Teaching Init Range: ({q_init[0]}, {q_init[1]})'
            save_dir = main_dir
            try:
                os.mkdir(save_dir)
            except:
                pass
            if plot_step:
                save_loc_step = save_dir + (f'nt_step.jpg')
                plot_step_progress(x_axis, save_loc_step, title,
                                   avg_steps_all_ep_nt, nt_label)
        if show_plot:
            plt.show()
        else:
            plt.close()

    teaching = True
    no_teaching = False

    for grid in grid_list:
        main_dir = f'{grid}_avg_{avg_over}_random_{init_random}/'
        try:
            os.mkdir(main_dir)
        except:
            pass

        env = gym.make(grid)
        num_actions = env.action_space.n
        num_orientations = 4
        actual_grid_height = env.height - 2
        actual_grid_width = env.width - 2
        terminal_states = [(actual_grid_width, actual_grid_height)]
        goal_state = terminal_states[-1]
        for theta in theta_list:
            for q_init in q_init_list:
                title = f'{grid}, Avg over {avg_over} runs, Theta = {theta}, ' \
                        f'Random = {init_random}'
                if no_teaching:
                    avg_steps_all_ep_nt = np.zeros(num_episodes // ep_chunk)
                    for avg_num in range(1, avg_over + 1):
                        print(
                            f'No Teaching {grid} Cons Init ({q_init[0]}, {q_init[1]}) '
                            f'run: {avg_num}')
                        q_table_nt = init_q_table(actual_grid_height,
                                                  actual_grid_width,
                                                  num_actions, random=init_random,
                                                  min=q_init[0],
                                                  max=q_init[1],
                                                  terminal_states=terminal_states)
                        steps_all_ep_nt, rew_all_ep_nt, t_rew_all_ep_nt = \
                            q_learning(
                                q_table_nt,
                                num_episodes,
                                max_steps_per_episode,
                                learning_rate,
                                discount_rate)

                        avg_steps_all_ep_nt += steps_all_ep_nt / avg_over

                        # policy_nt = get_optimal_policy(q_table_nt,
                        #                                   terminal_states=terminal_states)
                        # pretty_print_policy(policy_nt)

                if teaching:
                    avg_steps_all_ep_t = np.zeros(num_episodes // ep_chunk)
                    for avg_num in range(1, avg_over + 1):
                        print(
                            f'Teaching {grid}, Theta = {theta} Cons Init ({q_init[0]}, {q_init[1]}) run:'
                            f' {avg_num}')
                        q_table_t = init_q_table(actual_grid_height,
                                                 actual_grid_width,
                                                 num_actions, random=init_random,
                                                 min=q_init[0],
                                                 max=q_init[1],
                                                 terminal_states=terminal_states)
                        target_policy = gen_optimal_policy(actual_grid_height,
                                                           actual_grid_width,
                                                           goal_state)
                        q_table_t = teach(q_table_t, target_policy, theta=theta)
                        steps_all_ep_t, rew_all_ep_t, t_rew_all_ep_t = \
                            q_learning(
                                q_table_t,
                                num_episodes,
                                max_steps_per_episode,
                                learning_rate,
                                discount_rate)

                        avg_steps_all_ep_t += steps_all_ep_t / avg_over

                        # policy_t = get_optimal_policy(q_table_t,
                        #                                terminal_states=terminal_states)
                        # pretty_print_policy(policy_t)

                nt_label = f'No Teaching Init Range: ({q_init[0]}, {q_init[1]})'
                t_label = f'Teaching Init Range: ({q_init[0]}, {q_init[1]})'
                save_dir = main_dir
                try:
                    os.mkdir(save_dir)
                except:
                    pass
                save_loc_rew = save_dir + (f'rew_theta_{theta}.jpg')
                if plot_rew:
                    plot_all_reward_progress(x_axis, save_loc_rew, title,
                                             rew_all_ep_nt,
                                             t_rew_all_ep_nt, nt_label,
                                             rew_all_ep_t, t_rew_all_ep_t,
                                             t_label)
                if plot_step:
                    if plot_step:
                        if teaching and no_teaching:
                            save_loc_step = save_dir + (f't_nt_step_theta'
                                                        f'_{theta}.jpg')
                            plot_step_progress(x_axis, save_loc_step, title,
                                               avg_steps_all_ep_nt, nt_label,
                                               avg_steps_all_ep_t, t_label)
                        elif teaching:
                            save_loc_step = save_dir + (f't_step_theta'
                                                        f'_{theta}.jpg')
                            plot_step_progress(x_axis, save_loc_step, title,
                                               avg_steps_all_ep_t, t_label)
                        else:
                            save_loc_step = save_dir + (f'nt_step.jpg')
                            plot_step_progress(x_axis, save_loc_step, title,
                                               avg_steps_all_ep_nt, nt_label)
            if show_plot:
                plt.show()
            else:
                plt.close()
    exit()



    # Setting parameters
    ep_chunk = 5
    max_steps_per_episode = 100

    learning_rate = 0.1
    discount_rate = 0.9
    exploration_rate = 0.1

    num_episodes = 500
    theta_list = [0.15] * 1
    x_axis = np.linspace(1, num_episodes, int(num_episodes / ep_chunk))
    show_plot_rew = [False, False, False]
    show_plot_step = [True, True, True]
    q_init = [(10, 10), (0, 0), (-10, -10)]
    avg_over = 10
    run_list = [1, 1, 1]
    num_sub_optimal_runs = [0, 0, 0]
    theta_count = 0
    main_dir = grid + '/'
    try:
        os.mkdir(main_dir)
    except:
        pass

    for theta in theta_list:
        if theta_count % 20 == 0:
            print(f'On run: {theta_count}')
        theta_count += 1
        # print(f'theta is: {theta}')
        title = f'theta = {theta}'
        if run_list[0]:
            avg_steps_all_ep_nt_op = np.zeros(num_episodes // ep_chunk)
            for avg_num in range(1, avg_over + 1):
                print(f'No Teaching Cons Init Optimistic Q-learner run: '
                      f'{avg_num}')
                q_table_nt_op = init_q_table(actual_grid_height, actual_grid_width,
                                             num_actions, random=False,
                                             min=q_init[0][0],
                                             max=q_init[0][1],
                                             terminal_states=terminal_states)
                steps_all_ep_nt_op, rew_all_ep_nt_op, t_rew_all_ep_nt_op = \
                    q_learning(
                        q_table_nt_op,
                        num_episodes,
                        max_steps_per_episode,
                        learning_rate,
                        discount_rate)

                avg_steps_all_ep_nt_op += steps_all_ep_nt_op / avg_over

                # policy_nt_op = get_optimal_policy(q_table_nt_op,
                #                                   terminal_states=terminal_states)
                # pretty_print_policy(policy_nt_op)

            avg_steps_all_ep_t_op = np.zeros(num_episodes // ep_chunk)
            for avg_num in range(1, avg_over + 1):
                print(f'Teaching Cons Init Optimistic Q-learner run: '
                      f'{avg_num}')
                q_table_t_op = init_q_table(actual_grid_height, actual_grid_width,
                                            num_actions, random=False,
                                            min=q_init[0][0],
                                            max=q_init[0][1],
                                            terminal_states=terminal_states)
                target_policy_op = gen_optimal_policy(actual_grid_height,
                                                      actual_grid_width,
                                                      goal_state)
                q_table_t_op = teach(q_table_t_op, target_policy_op, theta=theta)
                steps_all_ep_t_op, rew_all_ep_t_op, t_rew_all_ep_t_op = \
                    q_learning(
                        q_table_t_op,
                        num_episodes,
                        max_steps_per_episode,
                        learning_rate,
                        discount_rate)

                avg_steps_all_ep_t_op += steps_all_ep_t_op / avg_over

                # policy_op = get_optimal_policy(q_table_t_op,
                #                                terminal_states=terminal_states)
                # pretty_print_policy(policy_op)

            nt_label_op = 'Optimistic No Teaching'
            t_label_op = 'Optimistic Teaching'
            save_dir = main_dir + 'optimistic_cons_init/'
            try:
                os.mkdir(save_dir)
            except:
                pass
            save_loc_rew = save_dir + (f'rew_optimistic_theta_{theta}.jpg')
            save_loc_step = save_dir + (f'step_optimistic_theta_{theta}.jpg')
            if show_plot_rew[0]:
                plot_all_reward_progress(x_axis, save_loc_rew, title,
                                         rew_all_ep_nt_op,
                                         t_rew_all_ep_nt_op, nt_label_op,
                                         rew_all_ep_t_op, t_rew_all_ep_t_op,
                                         t_label_op)
            if show_plot_step[0]:
                plot_step_progress(x_axis, save_loc_step, title,
                                   avg_steps_all_ep_nt_op, nt_label_op,
                                   avg_steps_all_ep_t_op, t_label_op)
            policy_pe = get_optimal_policy(q_table_t_op,
                                           terminal_states=terminal_states)
            num_sub_optimal_runs[0] += not check_if_optimal_policy(policy_pe)

        if run_list[1]:
            avg_steps_all_ep_nt_reg = np.zeros(num_episodes // ep_chunk)
            for avg_num in range(1, avg_over + 1):
                print(f'No Teaching Cons Init Regular Q-learner run: '
                      f'{avg_num}')
                q_table_nt_reg = init_q_table(actual_grid_height, actual_grid_width,
                                              num_actions, random=False,
                                              min=q_init[1][0],
                                              max=q_init[1][1],
                                              terminal_states=terminal_states)
                steps_all_ep_nt_reg, rew_all_ep_nt_reg, t_rew_all_ep_nt_reg = \
                    q_learning(
                        q_table_nt_reg,
                        num_episodes,
                        max_steps_per_episode,
                        learning_rate,
                        discount_rate)

                avg_steps_all_ep_nt_reg += steps_all_ep_nt_reg / avg_over

                # policy_nt_reg = get_optimal_policy(q_table_nt_reg,
                #                                    terminal_states=terminal_states)
                # pretty_print_policy(policy_nt_reg)

            avg_steps_all_ep_t_reg = np.zeros(num_episodes // ep_chunk)
            for avg_num in range(1, avg_over + 1):
                print(f'Teaching Cons Init Regular Q-learner run: '
                      f'{avg_num}')
                q_table_t_reg = init_q_table(actual_grid_height, actual_grid_width,
                                             num_actions, random=False,
                                             min=q_init[1][0],
                                             max=q_init[1][1],
                                             terminal_states=terminal_states)
                target_policy_reg = gen_optimal_policy(actual_grid_height,
                                                       actual_grid_width,
                                                       goal_state)
                q_table_t_reg = teach(q_table_t_reg, target_policy_reg, theta=theta)
                steps_all_ep_t_reg, rew_all_ep_t_reg, t_rew_all_ep_t_reg = \
                    q_learning(
                    q_table_t_reg,
                    num_episodes,
                    max_steps_per_episode,
                    learning_rate,
                    discount_rate)

                avg_steps_all_ep_t_reg += steps_all_ep_t_reg / avg_over

                # policy_reg = get_optimal_policy(q_table_t_reg,
                #                                 terminal_states=terminal_states)
                # pretty_print_policy(policy_reg)

            nt_label_reg = 'Regular No Teaching'
            t_label_reg = 'Regular Teaching'
            save_dir =  main_dir + 'regular_cons_init/'
            try:
                os.mkdir(save_dir)
            except:
                pass
            save_loc_rew = save_dir + (f'rew_regular_theta_{theta}.jpg')
            save_loc_step = save_dir + (f'step_regular_theta_{theta}.jpg')
            if show_plot_rew[1]:
                plot_all_reward_progress(x_axis, save_loc_rew, title,
                                         rew_all_ep_nt_reg,
                                         t_rew_all_ep_nt_reg, nt_label_reg,
                                         rew_all_ep_t_reg, t_rew_all_ep_t_reg,
                                         t_label_reg)
            if show_plot_step[1]:
                plot_step_progress(x_axis, save_loc_step, title,
                                   avg_steps_all_ep_nt_reg, nt_label_reg,
                                   avg_steps_all_ep_t_reg, t_label_reg)
            policy_pe = get_optimal_policy(q_table_t_reg,
                                           terminal_states=terminal_states)
            # num_sub_optimal_pe_runs[1] += not check_if_optimal_policy(policy_pe)
            num_sub_optimal_runs[1] += not min(t_rew_all_ep_t_reg)

        if run_list[2]:
            avg_steps_all_ep_nt_pe = np.zeros(num_episodes // ep_chunk)
            for avg_num in range(1, avg_over + 1):
                print(f'No Teaching Cons Init Pessimistic Q-learner run: '
                      f'{avg_num}')
                q_table_nt_pe = init_q_table(actual_grid_height, actual_grid_width,
                                             num_actions, random=False,
                                             min=q_init[2][0],
                                             max=q_init[2][1],
                                             terminal_states=terminal_states)
                steps_all_ep_nt_pe, rew_all_ep_nt_pe, t_rew_all_ep_nt_pe = q_learning(
                    q_table_nt_pe,
                    num_episodes,
                    max_steps_per_episode,
                    learning_rate,
                    discount_rate)

                avg_steps_all_ep_nt_pe += steps_all_ep_nt_pe / avg_over

                # policy_nt_pe = get_optimal_policy(q_table_nt_pe,
                #                                   terminal_states=terminal_states)
                # pretty_print_policy(policy_nt_pe)

            avg_steps_all_ep_t_pe = np.zeros(num_episodes // ep_chunk)
            for avg_num in range(1, avg_over + 1):
                print(f'Teaching Cons Init Pessimistic Q-learner run: '
                      f'{avg_num}')
                q_table_t_pe = init_q_table(actual_grid_height, actual_grid_width,
                                            num_actions, random=False,
                                            min=q_init[2][0],
                                            max=q_init[2][1],
                                            terminal_states=terminal_states)
                target_policy_pe = gen_optimal_policy(actual_grid_height,
                                                      actual_grid_width,
                                                      goal_state)
                q_table_t_pe = teach(q_table_t_pe, target_policy_pe, theta=theta)
                steps_all_ep_t_pe, rew_all_ep_t_pe, t_rew_all_ep_t_pe = \
                    q_learning(
                        q_table_t_pe,
                        num_episodes,
                        max_steps_per_episode,
                        learning_rate,
                        discount_rate)

                avg_steps_all_ep_t_pe += steps_all_ep_t_pe / avg_over

                # policy_pe = get_optimal_policy(q_table_t_pe,
                #                                terminal_states=terminal_states)
                # pretty_print_policy(policy_pe)

            nt_label_pe = 'Pessimistic No Teaching'
            t_label_pe = 'Pessimistic Teaching'
            save_dir = main_dir + 'pessimistic_cons_init/'
            try:
                os.mkdir(save_dir)
            except:
                pass
            save_loc_rew = save_dir + (f'rew_pessimistic_theta_{theta}.jpg')
            save_loc_step = save_dir + (f'step_pessimistic_theta_{theta}.jpg')
            if show_plot_rew[2]:
                plot_all_reward_progress(x_axis, save_loc_rew,
                                         title, rew_all_ep_nt_pe,
                                         t_rew_all_ep_nt_pe,
                                         nt_label_pe,
                                         rew_all_ep_t_pe, t_rew_all_ep_t_pe,
                                         t_label_pe)
            if show_plot_step[2]:
                plot_step_progress(x_axis, save_loc_step, title,
                                   steps_all_ep_nt_pe, nt_label_pe,
                                   steps_all_ep_t_pe, t_label_pe)

            policy_pe = get_optimal_policy(q_table_t_pe,
                                           terminal_states=terminal_states)
            num_sub_optimal_runs[2] += not check_if_optimal_policy(policy_pe)
            # num_sub_optimal_pe_runs += not min(t_rew_all_ep_t_pe)

        # print('#########DONE###########')
        plt.show()

        op_label = 'Optimistic'
        reg_label = 'Regular'
        pe_label = 'Pessimistic'
        # plot_all_reward_progress(x_axis, False, None, title,
        #     rew_all_ep_nt_op, t_rew_all_ep_nt_op, nt_label_op,
        #     rew_all_ep_nt_op, rew_all_ep_t_op, t_label_op,
        #     rew_all_ep_nt_reg, t_rew_all_ep_nt_reg, nt_label_reg,
        #     rew_all_ep_nt_reg, rew_all_ep_t_reg, t_label_reg,
        #     rew_all_ep_nt_pe, t_rew_all_ep_nt_pe, nt_label_pe,
        #     rew_all_ep_nt_pe, rew_all_ep_t_pe, t_label_pe)
        # plot_reward_progress_overlay(x_axis, rew_all_ep_t_op,
        #                              t_rew_all_ep_t_op, op_label,
        #                              rew_all_ep_t_reg, t_rew_all_ep_t_reg,
        #                              reg_label,
        #                              rew_all_ep_t_pe, t_rew_all_ep_t_pe,
        #                              pe_label)

    print(f'Number of sub-optimal pessimistic runs: {num_sub_optimal_runs}')

    exit()

    num_episodes = 100
    theta_list = [0.25, 0.15, 0.05]
    x_axis = np.linspace(ep_chunk, num_episodes, int(num_episodes / ep_chunk))
    save_plot = False
    for theta in theta_list:
        print(f'theta is: {theta}')
        print('No Teaching Rand Init Optimistic Q-learner')
        q_table_nt_op = init_q_table(actual_grid_height, actual_grid_width,
                                     num_actions, random=True, min=2, max=3,
                                     terminal_states=terminal_states)
        rew_all_ep_nt_op, t_rew_all_ep_nt_op, policy_changes_nt_op = q_learning(
            q_table_nt_op,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate)
        policy_nt_op = get_optimal_policy(q_table_nt_op,
                                          terminal_states=terminal_states)
        pretty_print_policy(policy_nt_op)

        print('Teaching Rand Init Optimistic Q-learner')
        q_table_t_op = init_q_table(actual_grid_height, actual_grid_width,
                                    num_actions, random=True, min=2, max=3,
                                    terminal_states=terminal_states)
        target_policy_op = gen_optimal_policy(actual_grid_height,
                                              actual_grid_width,
                                              goal_state)
        q_table_t_op = teach(q_table_t_op, target_policy_op, theta=theta)
        rew_all_ep_t_op, t_rew_all_ep_t_op, policy_changes_t_op = q_learning(
            q_table_t_op,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate)
        policy_op = get_optimal_policy(q_table_t_op,
                                       terminal_states=terminal_states)
        pretty_print_policy(policy_op)

        nt_label_op = 'Optimistic No Teaching'
        t_label_op = 'Optimistic Teaching'
        save_dir = 'optimistic_rand_init/'
        try:
            os.mkdir(save_dir)
        except:
            pass
        save_loc = save_dir + (f'optimistic_theta_{theta}.jpg')
        title = f'theta = {theta}'
        plot_all_reward_progress(x_axis, save_plot, save_loc, title,
                                 rew_all_ep_nt_op,
                                 t_rew_all_ep_nt_op, nt_label_op,
                                 rew_all_ep_t_op, t_rew_all_ep_t_op,
                                 t_label_op)

        print('No Teaching Rand Init Regular Q-learner')
        q_table_nt_reg = init_q_table(actual_grid_height, actual_grid_width,
                                      num_actions, random=True, min=0, max=1,
                                      terminal_states=terminal_states)
        rew_all_ep_nt_reg, t_rew_all_ep_nt_reg, policy_changes_nt_reg = \
            q_learning(
                q_table_nt_reg,
                num_episodes,
                max_steps_per_episode,
                learning_rate,
                discount_rate)
        policy_nt_reg = get_optimal_policy(q_table_nt_reg,
                                           terminal_states=terminal_states)
        pretty_print_policy(policy_nt_reg)

        print('Teaching Rand Init Regular Q-learner')
        q_table_t_reg = init_q_table(actual_grid_height, actual_grid_width,
                                     num_actions, random=True, min=0, max=1,
                                     terminal_states=terminal_states)
        target_policy_reg = gen_optimal_policy(actual_grid_height,
                                               actual_grid_width,
                                               goal_state)
        q_table_t_reg = teach(q_table_t_reg, target_policy_reg, theta=theta)
        rew_all_ep_t_reg, t_rew_all_ep_t_reg, policy_changes_t_reg = q_learning(
            q_table_t_reg,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate)
        policy_reg = get_optimal_policy(q_table_t_reg,
                                        terminal_states=terminal_states)
        pretty_print_policy(policy_reg)

        nt_label_reg = 'Regular No Teaching'
        t_label_reg = 'Regular Teaching'
        save_dir = 'regular_rand_init/'
        try:
            os.mkdir(save_dir)
        except:
            pass
        save_loc = save_dir + (f'regular_theta_{theta}.jpg')
        plot_all_reward_progress(x_axis, save_plot, save_loc, title,
                                 rew_all_ep_nt_reg,
                                 t_rew_all_ep_nt_reg, nt_label_reg,
                                 rew_all_ep_t_reg, t_rew_all_ep_t_reg,
                                 t_label_reg)

        print('No Teaching Rand Init Pessimistic Q-learner')
        q_table_nt_pe = init_q_table(actual_grid_height, actual_grid_width,
                                     num_actions, random=True, min=-3, max=-2,
                                     terminal_states=terminal_states)
        rew_all_ep_nt_pe, t_rew_all_ep_nt_pe, policy_changes_nt_pe = q_learning(
            q_table_nt_pe,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate)
        policy_nt_pe = get_optimal_policy(q_table_nt_pe,
                                          terminal_states=terminal_states)
        pretty_print_policy(policy_nt_pe)

        print('Teaching Rand Init Pessimistic Q-learner')
        q_table_t_pe = init_q_table(actual_grid_height, actual_grid_width,
                                    num_actions, random=True, min=-3, max=-2,
                                    terminal_states=terminal_states)
        target_policy_pe = gen_optimal_policy(actual_grid_height,
                                              actual_grid_width,
                                              goal_state)
        q_table_t_pe = teach(q_table_t_pe, target_policy_pe, theta=theta)
        rew_all_ep_t_pe, t_rew_all_ep_t_pe, policy_changes_t_pe = q_learning(
            q_table_t_pe,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate)
        policy_pe = get_optimal_policy(q_table_t_pe,
                                       terminal_states=terminal_states)
        pretty_print_policy(policy_pe)

        nt_label_pe = 'Pessimistic No Teaching'
        t_label_pe = 'Pessimistic Teaching'
        save_dir = 'pessimistic_rand_init/'
        try:
            os.mkdir(save_dir)
        except:
            pass
        save_loc = save_dir + (f'pessimistic_theta_{theta}.jpg')
        plot_all_reward_progress(x_axis, save_plot, save_loc,
                                 title, rew_all_ep_nt_pe,
                                 t_rew_all_ep_nt_pe, nt_label_pe,
                                 rew_all_ep_t_pe, t_rew_all_ep_t_pe,
                                 t_label_pe)

        print('#########DONE###########')

        op_label = 'Optimistic'
        reg_label = 'Regular'
        pe_label = 'Pessimistic'
        # plot_all_reward_progress(x_axis, False, None, title,
        #     rew_all_ep_nt_op, t_rew_all_ep_nt_op, nt_label_op,
        #     rew_all_ep_nt_op, rew_all_ep_t_op, t_label_op,
        #     rew_all_ep_nt_reg, t_rew_all_ep_nt_reg, nt_label_reg,
        #     rew_all_ep_nt_reg, rew_all_ep_t_reg, t_label_reg,
        #     rew_all_ep_nt_pe, t_rew_all_ep_nt_pe, nt_label_pe,
        #     rew_all_ep_nt_pe, rew_all_ep_t_pe, t_label_pe)
        # plot_reward_progress_overlay(x_axis, rew_all_ep_t_op,
        #                              t_rew_all_ep_t_op, op_label,
        #                              rew_all_ep_t_reg, t_rew_all_ep_t_reg,
        #                              reg_label,
        #                              rew_all_ep_t_pe, t_rew_all_ep_t_pe,
        #                              pe_label)

    exit()

    num_episodes = 100
    theta = 1
    x_axis = np.linspace(ep_chunk, num_episodes, int(num_episodes / ep_chunk))
    for i in range(1, 8):
        theta = theta / i
        print(f'theta is: {theta}')
        print('Teaching Rand Init Optimistic Q-learner')
        q_table_t_op = init_q_table(actual_grid_height, actual_grid_width,
                                    num_actions, random=True, min=2, max=3,
                                    terminal_states=terminal_states)
        target_policy_op = gen_optimal_policy(actual_grid_height,
                                              actual_grid_width,
                                              goal_state)
        q_table_t_op = teach(q_table_t_op, target_policy_op, theta=theta)
        rew_all_ep_t_op, t_rew_all_ep_t_op, policy_changes_t_op = q_learning(
            q_table_t_op,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate)
        policy_op = get_optimal_policy(q_table_t_op)
        pretty_print_policy(policy_op, terminal_states=terminal_states)

        print('Teaching Rand Init Regular Q-learner')
        q_table_t_reg = init_q_table(actual_grid_height, actual_grid_width,
                                     num_actions, random=True, min=0, max=1,
                                     terminal_states=terminal_states)
        target_policy_reg = gen_optimal_policy(actual_grid_height,
                                               actual_grid_width,
                                               goal_state)
        q_table_t_reg = teach(q_table_t_reg, target_policy_reg, theta=theta)
        rew_all_ep_t_reg, t_rew_all_ep_t_reg, policy_changes_t_reg = q_learning(
            q_table_t_reg,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate)
        policy_reg = get_optimal_policy(q_table_t_reg)
        pretty_print_policy(policy_reg, terminal_states=terminal_states)

        print('Teaching Rand Init Pessimistic Q-learner')
        q_table_t_pe = init_q_table(actual_grid_height, actual_grid_width,
                                    num_actions, random=True, min=-3, max=-2,
                                    terminal_states=terminal_states)
        target_policy_pe = gen_optimal_policy(actual_grid_height,
                                              actual_grid_width,
                                              goal_state)
        q_table_t_pe = teach(q_table_t_pe, target_policy_pe, theta=theta)
        rew_all_ep_t_pe, t_rew_all_ep_t_pe, policy_changes_t_pe = q_learning(
            q_table_t_pe,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate)
        policy_pe = get_optimal_policy(q_table_t_pe)
        pretty_print_policy(policy_pe, terminal_states=terminal_states)

        print('#########DONE###########')

        op_label = 'Optimistic'
        reg_label = 'Regular'
        pe_label = 'Pessimistic'
        plot_reward_progress_overlay(x_axis, rew_all_ep_t_op,
                                     t_rew_all_ep_t_op, op_label,
                                     rew_all_ep_t_reg, t_rew_all_ep_t_reg,
                                     reg_label,
                                     rew_all_ep_t_pe, t_rew_all_ep_t_pe,
                                     pe_label)

    exit()

    # Training
    print('Rand Init Optimistic Q-learner')
    q_table_nt_op = init_q_table(actual_grid_height, actual_grid_width,
                                 num_actions, random=True, min=1, max=2)
    rew_all_ep_nt_op, t_rew_all_ep_nt_op, policy_changes_nt_op = q_learning(
        q_table_nt_op,
        num_episodes,
        max_steps_per_episode,
        learning_rate,
        discount_rate)
    policy_nt_op = get_optimal_policy(q_table_nt_op, terminal_states)
    pretty_print_policy(policy_nt_op)

    print('Rand Init Pessimistic Q-learner')
    q_table_nt_pe = init_q_table(actual_grid_height, actual_grid_width,
                                 num_actions, random=True, min=-2, max=-1)
    rew_all_ep_nt_pe, t_rew_all_ep_nt_pe, policy_changes_nt_pe = q_learning(
        q_table_nt_pe,
        num_episodes,
        max_steps_per_episode,
        learning_rate,
        discount_rate)
    policy_nt_pe = get_optimal_policy(q_table_nt_pe, terminal_states)
    pretty_print_policy(policy_nt_pe)

    print('Teaching Rand Init Optimistic Q-learner')
    q_table_t_op = init_q_table(actual_grid_height, actual_grid_width,
                                num_actions, random=True, min=1, max=2)
    target_policy_op = gen_optimal_policy(actual_grid_height, actual_grid_width,
                                          goal_state)
    q_table_t_op = teach(q_table_t_op, target_policy_op, theta=0.2)
    rew_all_ep_t_op, t_rew_all_ep_t_op, policy_changes_t_op = q_learning(
        q_table_t_op,
        num_episodes,
        max_steps_per_episode,
        learning_rate,
        discount_rate)
    policy_t_op = get_optimal_policy(q_table_t_op, terminal_states)
    pretty_print_policy(policy_t_op)

    print('Teaching Rand Init Pessimistic Q-learner')
    q_table_t_pe = init_q_table(actual_grid_height, actual_grid_width,
                                num_actions, random=True, min=-2, max=-1)
    target_policy_pe = gen_optimal_policy(actual_grid_height, actual_grid_width,
                                          goal_state)
    q_table_t_pe = teach(q_table_t_pe, target_policy_pe, theta=0.2)
    rew_all_ep_t_pe, t_rew_all_ep_t_pe, policy_changes_t_pe = q_learning(
        q_table_t_pe,
        num_episodes,
        max_steps_per_episode,
        learning_rate,
        discount_rate)
    policy_t_pe = get_optimal_policy(q_table_t_pe, terminal_states)
    pretty_print_policy(policy_t_pe)

    print('############Plotting#################')
    x_axis = np.linspace(ep_chunk, num_episodes, int(num_episodes / ep_chunk))

    plot_reward_progress(x_axis, rew_all_ep_nt_op, t_rew_all_ep_nt_op)
    plot_reward_progress(x_axis, rew_all_ep_nt_pe, t_rew_all_ep_nt_pe)
    plot_reward_progress_overlay(x_axis, rew_all_ep_nt_op, rew_all_ep_nt_op,
                                 rew_all_ep_nt_pe, t_rew_all_ep_nt_pe)

    plot_reward_progress(x_axis, rew_all_ep_t_op, t_rew_all_ep_t_op)
    plot_reward_progress(x_axis, rew_all_ep_t_pe, t_rew_all_ep_t_pe)
    plot_reward_progress_overlay(x_axis, rew_all_ep_t_op, rew_all_ep_t_op,
                                 rew_all_ep_t_pe, t_rew_all_ep_t_pe)

    # Actually render environment and watch the agent
    # visualize_agent(q_table_nt_op, num_episodes=2)

    env.close()
