import numpy as np
import pandas as pd
import gym
import gym_minigrid


def get_optimal_value(num_states, state_num, action, step, horizon):
    distance_to_reward = num_states - state_num - 1
    optimal_value = horizon - step + 1  # if you could get reward each step
    if distance_to_reward == 0:
        optimal_value -= action
    else:
        optimal_value -= distance_to_reward - 1  # need to get to reward
        if state_num == 0:
            optimal_value -= action
        else:
            optimal_value -= 2 * action
    return max(optimal_value, 0)


def np_init_optimal_values(num_states, num_actions, step, horizon):
    init_values = np.zeros((num_states, num_actions))
    for state_num in range(num_states):
        for action in range(num_actions):
            init_values[state_num, action] = get_optimal_value(
                num_states, state_num, action, step, horizon)
    return init_values


def init_optimal_q_table(env, horizon):
    num_actions = env.action_space.n
    grid_height = env.height - 2  # grid has padding of 1
    grid_width = env.width - 2  # grid has padding of 1
    states = []
    for a in range(1, grid_height + 1):
        for b in range(1, grid_width + 1):
            states.append((b, a))

    q_table = {}
    for step in range(1, horizon + 1):
        init_values = np_init_optimal_values(len(states), num_actions, step, horizon)
        q_table_wo_hrzn = pd.DataFrame(init_values,
                                       index=pd.MultiIndex.from_tuples(states),
                                       columns=np.linspace(
                                           0, num_actions - 1,
                                           num_actions,
                                           dtype=int))
        q_table[step] = q_table_wo_hrzn
    return q_table


if __name__ == "__main__":
    horizon = 3
    grid = 'MiniGrid-Empty-1x3-v0'
    env = gym.make(grid)
    optimal_q_table = init_optimal_q_table(env, horizon)
    pass
