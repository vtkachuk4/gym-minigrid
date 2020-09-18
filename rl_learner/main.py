import numpy as np
import pandas as pd
import gym
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gym_minigrid


def test_cur_policy(q_table, step_reward, test_eps=500):
    avg_reward_all_episodes = 0
    for eps in range(test_eps):
        state = env.reset()

        done = False
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            action = np.argmax(q_table[state, :])

            new_state, reward, done, info = env.step(action)

            if step_reward is not None:
                if reward == 0:
                    reward = step_reward
                if new_state in [5, 7, 11, 12]:
                    reward = step_reward * 100

            state = new_state
            rewards_current_episode += reward

            if done == True:
                break

        avg_reward_all_episodes += rewards_current_episode / test_eps
    return avg_reward_all_episodes


def teach_lvl_one(target_policy):
    q_table = np.zeros((state_space_size, action_space_size))
    _epsilon = 0.1
    for s, a in target_policy:
        q_table[s, a] = np.max(q_table[s, :]) + _epsilon
    return q_table


def get_optimal_policy(q_table):
    optimal_policy = [q_table.columns[q_table.loc[row].argmax()] for row in
                      q_table.index]
    return optimal_policy


def print_optimal_policy(q_table, terminal_states=[5, 7, 11, 12, 15],
                         pretty=True):
    optimal_policy = get_optimal_policy(q_table)
    if pretty:
        arrow_dict = {0: '<', 1: 'v', 2: '>', 3: '^'}
        optimal_policy = [[arrow_dict[a] for a in row] for row in
                          optimal_policy]
    for terminal_state in terminal_states:
        row = terminal_state // 4
        column = terminal_state % 4
        optimal_policy[row][column] = '0'
    print(np.array(optimal_policy))


# Q-learning algorithm
def q_learning(q_table, num_episodes, max_steps_per_episode, learning_rate,
               discount_rate, step_reward=None):
    rewards_all_episodes = []
    test_reward_per_thousand_episodes = []
    policy_changes = 0
    new_policy = get_optimal_policy(q_table)
    print(new_policy)

    for episode in range(num_episodes):
        # if episode % ep_chunk == 0:
        #     test_reward_per_thousand_episodes.append(
        #         test_cur_policy(q_table, step_reward=step_reward))

        state = env.reset()

        done = False
        rewards_current_episode = 0
        # if episode == 19000:
        # exploration_rate = 0

        for step in range(max_steps_per_episode):
            # env.render()
            old_policy = new_policy

            # Exploration and Exploitation
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = q_table.columns[q_table.loc[state].argmax()]
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)
            # if step_reward is not None:
            #     if reward == 0:
            #         reward = step_reward
            #     if new_state in [5, 7, 11, 12]:
            #         reward = step_reward * 100

            # update Q-table
            q_table[action].loc[state] = q_table[action].loc[state] * (
                    1 - learning_rate) + learning_rate * (
                                                 reward + discount_rate *
                                                 q_table.loc[new_state].max())

            new_policy = get_optimal_policy(q_table)
            # if new_policy != old_policy:
            #     policy_changes += 1

            state = new_state
            rewards_current_episode += reward

            if done == True:
                break

        # Eploration rate decay
        # exploration_rate = min_exploration_rate + \
        #  (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

        rewards_all_episodes.append(rewards_current_episode)

    reward_per_thousand_episodes = np.split(np.array(rewards_all_episodes),
                                            num_episodes / ep_chunk)
    reward_per_thousand_episodes = [sum(r / ep_chunk) for r in
                                    reward_per_thousand_episodes]

    count = ep_chunk
    print("******Average Reward Per Thousand Episodes*****\n")
    for r in reward_per_thousand_episodes:
        print(count, ": ", r)
        count += ep_chunk

    count = ep_chunk
    for r in test_reward_per_thousand_episodes:
        print("test ", count, ": ", r)
        count += ep_chunk
    return test_reward_per_thousand_episodes, reward_per_thousand_episodes, \
           policy_changes


def plot_reward_progress(x_axis, y_axis_1, y_axis_2=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

    axes[0].plot(x_axis, y_axis_1)
    axes[0].set(xlabel='episodes', ylabel='Avg Total Reward')
    if y_axis_2 is not None:
        axes[1].plot(x_axis, y_axis_2)
        axes[1].set(xlabel='episodes', ylabel='Avg Total Reward')
    fig.tight_layout()


def plot_reward_progress_overlay(x_axis, y_axis_1_1, y_axis_2_1, y_axis_1_2,
                                 y_axis_2_2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    axes[0].plot(x_axis, y_axis_1_1, label='Zero Init')
    axes[0].plot(x_axis, y_axis_1_2, label='Teach Init')
    axes[0].set(xlabel='episodes', ylabel='Avg Total Reward')
    axes[1].plot(x_axis, y_axis_2_1, label='Zero Init')
    axes[1].plot(x_axis, y_axis_2_2, label='Teach Init')
    axes[1].set(xlabel='episodes', ylabel='Avg Total Reward')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()


def init_q_table(height, width, num_orientations, num_actions):
    states = []
    for a in range(1, height + 1):
        for b in range(1, width + 1):
            for c in range(num_orientations):
                states.append((a, b, c))

    # q_table = np.zeros((state_space_size, action_space_size))
    q_table = pd.DataFrame(np.zeros((len(states), num_actions)),
                           index=pd.MultiIndex.from_tuples(states),
                           columns=np.linspace(
                               0, num_actions - 1, num_actions, dtype=int))
    return q_table


if __name__ == '__main__':
    env = gym.make('MiniGrid-Empty-5x5-v0')

    num_actions = env.action_space.n
    a = env.action_space
    num_orientations = 4
    q_table = init_q_table(env.height, env.width, num_orientations, num_actions)
    print(q_table)

    # observation = env.reset()
    # for i in range(100):
    #     env.render()
    #     print(observation)
    #     observation, reward, done, info = env.step(env.action_space.sample())
    #     if done:
    #         break

    target_policy = [(0, 0), (1, 3), (2, 3), (3, 3),
                     (4, 0), (5, 0), (6, 0), (7, 0),
                     (8, 3), (9, 1), (10, 0), (11, 0),
                     (12, 0), (13, 2), (14, 1), (15, 0)]

    # q_table = teach_lvl_one(target_policy)
    # print_optimal_policy(q_table)

    num_episodes = 5
    ep_chunk = 5
    max_steps_per_episode = 100

    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 0.4

    print('Teaching 1st Q-learner')
    q_table_z = init_q_table(env.height, env.width, num_orientations,
                             num_actions)
    t_rew_all_ep_z, rew_all_ep_z, policy_changes = q_learning(q_table_z,
                                                              num_episodes,
                                                              max_steps_per_episode,
                                                              learning_rate,
                                                              discount_rate)
    # print_optimal_policy(q_table_z)
    print('policy changes: ', policy_changes)
    print(q_table_z)

    if False:
        print('Teaching 2nd Q-learner')
        q_table_t = teach_lvl_one(target_policy)
        t_rew_all_ep_t, rew_all_ep_t, policy_changes = q_learning(q_table_t,
                                                                  num_episodes,
                                                                  max_steps_per_episode,
                                                                  learning_rate,
                                                                  discount_rate)
        print_optimal_policy(q_table_t)
        print('policy changes: ', policy_changes)
        print(q_table_t)

        print('Teaching 3rd Q-learner')
        q_table_rz = np.zeros((state_space_size, action_space_size))
        t_rews_all_ep_rz, rews_all_ep_rz, policy_changes = q_learning(
            q_table_rz,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate,
            step_reward=-0.1)
        print_optimal_policy(q_table_rz)
        print('policy changes: ', policy_changes)
        print(q_table_rz)

        print('Teaching 4th Q-learner')
        q_table_rt = teach_lvl_one(target_policy)
        t_rews_all_ep_rt, rews_all_ep_rt, policy_changes = q_learning(
            q_table_rt,
            num_episodes,
            max_steps_per_episode,
            learning_rate,
            discount_rate,
            step_reward=-0.1)
        print_optimal_policy(q_table_rt)
        print('policy changes: ', policy_changes)
        print(q_table_rt)

    print('############Plotting#################3')
    x_axis = np.linspace(ep_chunk, num_episodes, int(num_episodes / ep_chunk))

    plot_reward_progress(x_axis, rew_all_ep_z)
    # plot_reward_progress(x_axis, rew_all_ep_z, t_rew_all_ep_z)
    # plot_reward_progress(x_axis, rew_all_ep_t, t_rew_all_ep_t)
    # plot_reward_progress_overlay(x_axis, rew_all_ep_z, t_rew_all_ep_z,
    #                              rew_all_ep_t, t_rew_all_ep_t)
    # plot_reward_progress(x_axis, rews_all_ep_rz, t_rews_all_ep_rz)
    # plot_reward_progress(x_axis, rews_all_ep_rt, t_rews_all_ep_rt)
    # plot_reward_progress_overlay(x_axis, rews_all_ep_rz, t_rews_all_ep_rz,
    #                              rews_all_ep_rt, t_rews_all_ep_rt)

    for episode in range(3):
        state = env.reset()
        # print(state)
        done = False
        print("*****Episode ", episode + 1, "*****\n\n\n\n")
        time.sleep(1)

        for step in range(max_steps_per_episode):
            clear_output(wait=True)
            env.render()
            time.sleep(0.3)

            action = q_table.columns[q_table_z.loc[state].argmax()]
            new_state, reward, done, info = env.step(action)

            if done:
                clear_output(wait=True)
                env.render()
                if reward == 1:
                    print("*****You've reached your goal!*****")
                    time.sleep(3)
                else:
                    print("*****You fell through a hole!*****")
                    time.sleep(3)
                clear_output(wait=True)
                break
            state = new_state

    env.close()
