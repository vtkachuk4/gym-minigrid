import numpy as np
import pandas as pd
import gym
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gym_minigrid


def test_cur_policy(q_table, test_eps=10):
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
            q_table.loc[state, target_action] = q_table.loc[state].max() + theta
    return q_table


def get_optimal_policy(q_table, terminal_states=None):
    height = q_table[0].loc[:, 1].size
    width = q_table[0].loc[1, :].size
    optimal_policy = pd.DataFrame(np.zeros((width, height)),
                           index=np.linspace(1, height, height, dtype=int),
                           columns=np.linspace(1, width, width, dtype=int))
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
                           index=np.linspace(1, height, height, dtype=int),
                           columns=np.linspace(1, width, width, dtype=int))
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            pretty_policy.loc[i, j] = arrow_dict[policy.loc[i, j]]
    print(pretty_policy)


# Q-learning algorithm
def q_learning(q_table, num_episodes, max_steps_per_episode, learning_rate,
               discount_rate, test_policy=True):
    rewards_all_episodes = []
    test_reward_per_thousand_episodes = []
    policy_changes = 0
    # new_policy = get_optimal_policy(q_table)
    # print(new_policy)

    for episode in range(num_episodes):
        # print(f'we are on episode: {episode}')
        if test_policy:
            if episode % ep_chunk == 0:
                test_reward_per_thousand_episodes.append(
                    test_cur_policy(q_table))

        state = env.reset()

        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            # env.render()
            # old_policy = new_policy

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

            # new_policy = get_optimal_policy(q_table)
            # if new_policy != old_policy:
            #     policy_changes += 1

            state = new_state
            rewards_current_episode += reward

            if done == True:
                break

        rewards_all_episodes.append(rewards_current_episode)
        if episode % ep_chunk == 0 and episode != 0:
            avg_rew = sum(rewards_all_episodes[episode - ep_chunk:episode]) / \
                      ep_chunk
            print(f'Episode: {episode}, avg reward for last {ep_chunk} '
                  f'episodes: {avg_rew}')

    reward_per_thousand_episodes = np.split(np.array(rewards_all_episodes),
                                            num_episodes / ep_chunk)
    reward_per_thousand_episodes = [sum(r / ep_chunk) for r in
                                    reward_per_thousand_episodes]

    return reward_per_thousand_episodes, test_reward_per_thousand_episodes, \
           policy_changes


def plot_reward_progress(x_axis, y_axis_1, y_axis_2=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

    axes[0].plot(x_axis, y_axis_1)
    axes[0].set(xlabel='episodes', ylabel='Avg Total Reward')
    if y_axis_2 is not None:
        axes[1].plot(x_axis, y_axis_2)
        axes[1].set(xlabel='episodes', ylabel='Avg Total Reward')
    fig.tight_layout()
    plt.show()


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
    plt.show()


def init_q_table(height, width, num_actions):
    states = []
    for a in range(1, height + 1):
        for b in range(1, width + 1):
            states.append((a, b))

    q_table = pd.DataFrame(np.zeros((len(states), num_actions)),
                           index=pd.MultiIndex.from_tuples(states),
                           columns=np.linspace(
                               0, num_actions - 1, num_actions, dtype=int))
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
                           index=np.linspace(1, height, height, dtype=int),
                           columns=np.linspace(1, width, width, dtype=int))
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            if j < width:
                optimal_policy.loc[i, j] = 0
            else:
                optimal_policy.loc[i, j] = 1
    optimal_policy.loc[goal_state[1], goal_state[0]] = 'NaN'
    return optimal_policy


if __name__ == '__main__':
    env = gym.make('MiniGrid-Empty-8x8-v0')

    num_actions = env.action_space.n
    num_orientations = 4
    actual_grid_height = env.height - 2
    actual_grid_width = env.width - 2
    terminal_states = [(actual_grid_width, actual_grid_height)]
    goal_state = terminal_states[-1]

    # Testing stuff out
    q_table_init = init_q_table(actual_grid_height, actual_grid_width, num_actions)
    print(q_table_init)
    policy_init = get_optimal_policy(q_table_init, terminal_states)
    print(policy_init)
    pretty_print_policy(policy_init)

    teaching_policy = gen_optimal_policy(actual_grid_height,
                                         actual_grid_width, goal_state)
    pretty_print_policy(teaching_policy)

    # Setting learning parameters
    num_episodes = 100
    ep_chunk = 10
    max_steps_per_episode = 100

    learning_rate = 0.1
    discount_rate = 0.99
    exploration_rate = 0.1

    # Training
    print('Teaching 1st Q-learner')
    q_table_nt = init_q_table(actual_grid_height, actual_grid_width,
                              num_actions)
    rew_all_ep_nt, t_rew_all_ep_nt, policy_changes_t = q_learning(q_table_nt,
                                                              num_episodes,
                                                              max_steps_per_episode,
                                                              learning_rate,
                                                              discount_rate)
    policy_nt = get_optimal_policy(q_table_nt, terminal_states)
    pretty_print_policy(policy_nt)

    print('Teaching 2nd Q-learner')
    q_table_t = init_q_table(actual_grid_height, actual_grid_width,
                              num_actions)
    target_policy = gen_optimal_policy(actual_grid_height, actual_grid_width,
                                       goal_state)
    q_table_t = teach(q_table_t, target_policy, theta=0.2)
    rew_all_ep_t, t_rew_all_ep_t, policy_changes_t = q_learning(q_table_t,
                                                              num_episodes,
                                                              max_steps_per_episode,
                                                              learning_rate,
                                                              discount_rate)
    policy_t = get_optimal_policy(q_table_t, terminal_states)
    pretty_print_policy(policy_t)

    print('############Plotting#################')
    x_axis = np.linspace(ep_chunk, num_episodes, int(num_episodes / ep_chunk))

    plot_reward_progress(x_axis, rew_all_ep_nt, t_rew_all_ep_nt)
    plot_reward_progress(x_axis, rew_all_ep_t, t_rew_all_ep_t)
    plot_reward_progress_overlay(x_axis, rew_all_ep_nt, rew_all_ep_nt,
                                 rew_all_ep_t, t_rew_all_ep_t)

    # Actually render environment and watch the agent
    visualize_agent(q_table_nt, num_episodes=2)

    env.close()
