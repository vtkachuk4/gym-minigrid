import pandas as pd
import numpy as np
import random
import time
from helpers import init_optimal_q_table


class EGreedyQLearner:
    def __init__(self, env, horizon, init_val=0):
        self.name = 'e-Greedy Q-learning'
        self.env = env
        self.horizon = horizon
        self.init_val = init_val
        self.num_actions = env.action_space.n
        self.grid_height = env.height - 2  # grid has padding of 1
        self.grid_width = env.width - 2  # grid has padding of 1
        self.q_table = self.init_q_table()
        self.visit_count = self.init_visit_count()

    def init_q_table(self):
        states = []
        for a in range(1, self.grid_height + 1):
            for b in range(1, self.grid_width + 1):
                states.append((b, a))
        init_vals = np.random.rand(len(states), self.num_actions) + \
                    self.init_val

        q_table_wo_hrzn = pd.DataFrame(init_vals,
                                       index=pd.MultiIndex.from_tuples(states),
                                       columns=np.linspace(
                                           0, self.num_actions - 1,
                                           self.num_actions,
                                           dtype=int))
        q_table = {1: q_table_wo_hrzn}
        for step in range(2, self.horizon + 1):
            q_table[step] = q_table_wo_hrzn.copy()
        return q_table

    def init_visit_count(self):
        states = []
        for a in range(1, self.grid_height + 1):
            for b in range(1, self.grid_width + 1):
                states.append((b, a))
        init_vals = np.zeros((len(states), self.num_actions))

        visit_count_wo_hrzn = pd.DataFrame(init_vals,
                                           index=pd.MultiIndex.from_tuples(
                                               states),
                                           columns=np.linspace(
                                               0, self.num_actions - 1,
                                               self.num_actions,
                                               dtype=int))
        visit_count = {1: visit_count_wo_hrzn}
        for step in range(2, self.horizon + 1):
            visit_count[step] = visit_count_wo_hrzn.copy()
        return visit_count

    def learn(self, num_episodes, exploration_rate=0.1, ep_chunk=1,
              print_rew=False):
        rewards = np.zeros(num_episodes)
        start_states = []

        for episode in range(1, num_episodes + 1):
            # print(f'we are on episode: {episode}')
            state = self.env.reset()
            start_states.append(state)
            rewards_current_episode = 0
            step = 1
            while True:
                # Exploration and Exploitation
                exploration_rate_threshold = random.uniform(0, 1)
                if exploration_rate_threshold > exploration_rate:
                    action = self.q_table[step].columns[self.q_table[step].loc[
                        state].argmax()]
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, info = self.env.step(action)
                self.visit_count[step].loc[state, action] += 1

                # update Q-table
                learning_rate = 1 / self.visit_count[step].loc[state,action]
                if step == self.horizon:
                    next_step_value = 0
                else:
                    next_step_value = self.q_table[step + 1].loc[new_state].max()
                td_error = reward + next_step_value - \
                           self.q_table[step].loc[state, action]
                self.q_table[step].loc[state, action] = self.q_table[step].loc[
                                                            state,
                                                            action] + \
                                                        learning_rate * (
                                                            td_error)

                state = new_state
                rewards_current_episode += reward

                if step == self.horizon:
                    break
                step += 1

            rewards[episode - 1] = rewards_current_episode

            if print_rew:
                if episode % ep_chunk == 0 and episode != 0:
                    avg_rew = sum(
                        rewards[episode - ep_chunk:episode]) / \
                              ep_chunk
                    print(
                        f'Episode: {episode}, avg reward for last {ep_chunk} '
                        f'episodes: {avg_rew}')
        return rewards, start_states

    def get_optimal_policy(self):
        optimal_policy_wo_hrzn = pd.DataFrame(np.zeros((self.grid_height,
                                                        self.grid_width)),
                                              index=np.linspace(1,
                                                                self.grid_height,
                                                                self.grid_height,
                                                                dtype=int),
                                              columns=np.linspace(1,
                                                                  self.grid_width,
                                                                  self.grid_width,
                                                                  dtype=int))
        optimal_policy = {}
        for step in range(1, self.horizon + 1):
            optimal_policy[step] = optimal_policy_wo_hrzn.copy()
            for i in range(1, self.grid_height + 1):
                for j in range(1, self.grid_width + 1):
                    # remembering indices for q_table are x, y values
                    optimal_policy[step].loc[i, j] = \
                        self.q_table[step].columns[self.q_table[step].loc[j,
                                                                          i].argmax()]
        return optimal_policy

    def pretty_print_policy(self, policy, step):
        assert step >= 1, 'step must be at least 1'
        # arrow_dict = {0: '>', 1: 'v', 2: '<', 3: '^', 'NaN': 0}
        arrow_dict = {0: '>', 1: '<', 'NaN': 0}
        pretty_policy = pd.DataFrame(np.zeros((self.grid_height,
                                               self.grid_width)),
                                     index=np.linspace(1, self.grid_height,
                                                       self.grid_height,
                                                       dtype=int),
                                     columns=np.linspace(1, self.grid_width,
                                                         self.grid_width,
                                                         dtype=int))
        for i in range(1, self.grid_height + 1):
            for j in range(1, self.grid_width + 1):
                pretty_policy.loc[i, j] = arrow_dict[policy[step].loc[i, j]]
        print(pretty_policy)

    def visualize_agent(self, num_episodes=1):
        for episode in range(num_episodes):
            state = self.env.reset()
            for step in range(1, self.horizon + 1):
                self.env.render()
                time.sleep(0.5)

                action = self.q_table[step].columns[self.q_table[step].loc[
                    state].argmax()]
                new_state, reward, done, info = self.env.step(action)
                state = new_state

                if step == self.horizon:
                    time.sleep(2)


class UCBHQLearner():
    def __init__(self, env, horizon):
        self.name = 'UCB-H Q-learning'
        self.env = env
        self.horizon = horizon
        self.num_actions = env.action_space.n
        self.grid_height = env.height - 2  # grid has padding of 1
        self.grid_width = env.width - 2  # grid has padding of 1
        self.num_states = self.grid_height * self.grid_width
        self.q_table = self.init_q_table()
        self.visit_count = self.init_visit_count()

    def init_q_table(self):
        states = []
        for a in range(1, self.grid_height + 1):
            for b in range(1, self.grid_width + 1):
                states.append((b, a))
        init_vals = np.zeros((len(states), self.num_actions)) + \
                    self.horizon

        q_table_wo_hrzn = pd.DataFrame(init_vals,
                                       index=pd.MultiIndex.from_tuples(states),
                                       columns=np.linspace(
                                           0, self.num_actions - 1,
                                           self.num_actions,
                                           dtype=int))
        q_table = {1: q_table_wo_hrzn}
        for step in range(2, self.horizon + 1):
            q_table[step] = q_table_wo_hrzn.copy()
        return q_table

    def init_visit_count(self):
        states = []
        for a in range(1, self.grid_height + 1):
            for b in range(1, self.grid_width + 1):
                states.append((b, a))
        init_vals = np.zeros((len(states), self.num_actions))

        visit_count_wo_hrzn = pd.DataFrame(init_vals,
                                           index=pd.MultiIndex.from_tuples(
                                               states),
                                           columns=np.linspace(
                                               0, self.num_actions - 1,
                                               self.num_actions,
                                               dtype=int))
        visit_count = {1: visit_count_wo_hrzn}
        for step in range(2, self.horizon + 1):
            visit_count[step] = visit_count_wo_hrzn.copy()
        return visit_count

    def learn(self, num_episodes, ep_chunk=1, c=0.1, p=0.05, print_rew=False):
        rewards = np.zeros(num_episodes)
        start_states = []

        for episode in range(1, num_episodes + 1):
            # print(f'we are on episode: {episode}')
            state = self.env.reset()
            start_states.append(state)
            rewards_current_episode = 0
            step = 1
            while True:
                # action = self.q_table[step].columns[self.q_table[step].loc[
                #     state].argmax()]
                actions = self.q_table[step].loc[state]
                list_max_action_indices = np.flatnonzero(actions == actions.max())
                max_action_index = np.random.choice(list_max_action_indices)
                action = self.q_table[step].columns[max_action_index]

                new_state, reward, done, info = self.env.step(action)
                self.visit_count[step].loc[state, action] += 1

                # update Q-table
                t = self.visit_count[step].loc[state, action]
                learning_rate = (self.horizon + 1) / (self.horizon + t)
                T = num_episodes * self.horizon
                iota = np.log(self.num_states * self.num_actions * T / p)
                bonus = c * np.sqrt(self.horizon ** 3 * iota / t)
                if step == self.horizon:
                    next_step_value = 0
                else:
                    next_step_value = self.q_table[step + 1].loc[new_state].max()
                td_error = reward + min(next_step_value, self.horizon) - self.q_table[step].loc[
                    state, action] + bonus
                self.q_table[step].loc[state, action] = self.q_table[
                                                            step].loc[state,
                                                                      action] + learning_rate * (
                                                            td_error)

                state = new_state
                rewards_current_episode += reward

                if step == self.horizon:
                    break
                step += 1

            rewards[episode - 1] = rewards_current_episode

            if print_rew:
                if episode % ep_chunk == 0 and episode != 0:
                    avg_rew = sum(
                        rewards[episode - ep_chunk:episode]) / \
                              ep_chunk
                    print(
                        f'Episode: {episode}, avg reward for last {ep_chunk} '
                        f'episodes: {avg_rew}')
        return rewards, start_states

    def get_optimal_policy(self):
        optimal_policy_wo_hrzn = pd.DataFrame(np.zeros((self.grid_height,
                                                        self.grid_width)),
                                              index=np.linspace(1,
                                                                self.grid_height,
                                                                self.grid_height,
                                                                dtype=int),
                                              columns=np.linspace(1,
                                                                  self.grid_width,
                                                                  self.grid_width,
                                                                  dtype=int))
        optimal_policy = {}
        for step in range(1, self.horizon + 1):
            optimal_policy[step] = optimal_policy_wo_hrzn.copy()
            for i in range(1, self.grid_height + 1):
                for j in range(1, self.grid_width + 1):
                    # remembering indices for q_table are x, y values
                    optimal_policy[step].loc[i, j] = \
                        self.q_table[step].columns[self.q_table[step].loc[j,
                                                                          i].argmax()]
        return optimal_policy

    def pretty_print_policy(self, policy, step):
        assert step >= 1, 'step must be at least 1'
        # arrow_dict = {0: '>', 1: 'v', 2: '<', 3: '^', 'NaN': 0}
        arrow_dict = {0: '>', 1: '<', 'NaN': 0}
        pretty_policy = pd.DataFrame(np.zeros((self.grid_height,
                                               self.grid_width)),
                                     index=np.linspace(1, self.grid_height,
                                                       self.grid_height,
                                                       dtype=int),
                                     columns=np.linspace(1, self.grid_width,
                                                         self.grid_width,
                                                         dtype=int))
        for i in range(1, self.grid_height + 1):
            for j in range(1, self.grid_width + 1):
                pretty_policy.loc[i, j] = arrow_dict[policy[step].loc[i, j]]
        print(pretty_policy)

    def visualize_agent(self, num_episodes=1):
        for episode in range(num_episodes):
            state = self.env.reset()
            for step in range(1, self.horizon + 1):
                self.env.render()
                time.sleep(0.5)

                action = self.q_table[step].columns[self.q_table[step].loc[
                    state].argmax()]
                new_state, reward, done, info = self.env.step(action)
                state = new_state

                if step == self.horizon:
                    time.sleep(2)

class UCBHQLearnerOI():
    def __init__(self, env, horizon, non_op_steps=[1], non_op_states=[(1,1)],
                 non_op_actions=[1]):
        self.name = 'UCB-H Q-learning Optimal Init'
        self.env = env
        self.horizon = horizon
        self.non_op_steps = non_op_steps
        self.non_op_states = non_op_states
        self.non_op_actions = non_op_actions
        self.num_actions = env.action_space.n
        self.grid_height = env.height - 2  # grid has padding of 1
        self.grid_width = env.width - 2  # grid has padding of 1
        self.num_states = self.grid_height * self.grid_width
        self.q_table = self.init_q_table()
        self.visit_count = self.init_visit_count()

    def init_q_table(self):
        q_table = init_optimal_q_table(self.env, self.horizon)
        for step in self.non_op_steps:
            for state in self.non_op_states:
                for action in self.non_op_actions:
                    q_table[step].loc[state, action] = self.horizon
        return q_table

    def init_visit_count(self):
        init_vals = np.zeros((len(self.non_op_states),
                              len(self.non_op_actions)))
        visit_count_wo_hrzn = pd.DataFrame(init_vals,
                                           index=pd.MultiIndex.from_tuples(
                                               self.non_op_states),
                                           columns=self.non_op_actions)
        visit_count = {}
        for step in self.non_op_steps:
            visit_count[step] = visit_count_wo_hrzn.copy()
        return visit_count

    def learn(self, num_episodes, ep_chunk=1, p=0.05, c=0.1,  print_rew=False):
        rewards = np.zeros(num_episodes)
        start_states = []

        for episode in range(1, num_episodes + 1):
            # print(f'we are on episode: {episode}')
            state = self.env.reset()
            start_states.append(state)
            rewards_current_episode = 0
            step = 1
            while True:
                # action = self.q_table[step].columns[self.q_table[step].loc[
                #     state].argmax()]
                actions = self.q_table[step].loc[state]
                list_max_action_indices = np.flatnonzero(actions == actions.max())
                max_action_index = np.random.choice(list_max_action_indices)
                action = self.q_table[step].columns[max_action_index]

                new_state, reward, done, info = self.env.step(action)

                if step in self.non_op_steps and state in self.non_op_states and action in self.non_op_actions:
                    self.visit_count[step].loc[state, action] += 1

                    # update Q-table
                    t = self.visit_count[step].loc[state, action]
                    learning_rate = (self.horizon + 1) / (self.horizon + t)
                    T = num_episodes * self.horizon
                    iota = np.log(self.num_states * self.num_actions * T / p)
                    bonus = c * np.sqrt(self.horizon ** 3 * iota / t)
                    if step == self.horizon:
                        next_step_value = 0
                    else:
                        next_step_value = self.q_table[step + 1].loc[new_state].max()
                    td_error = reward + min(next_step_value, self.horizon) - self.q_table[step].loc[
                        state, action] + bonus
                    self.q_table[step].loc[state, action] = self.q_table[
                                                                step].loc[state,
                                                                          action] + learning_rate * (
                                                                td_error)

                state = new_state
                rewards_current_episode += reward

                if step == self.horizon:
                    break
                step += 1

            rewards[episode - 1] = rewards_current_episode

            if print_rew:
                if episode % ep_chunk == 0 and episode != 0:
                    avg_rew = sum(
                        rewards[episode - ep_chunk:episode]) / \
                              ep_chunk
                    print(
                        f'Episode: {episode}, avg reward for last {ep_chunk} '
                        f'episodes: {avg_rew}')
        return rewards, start_states

    def get_optimal_policy(self):
        optimal_policy_wo_hrzn = pd.DataFrame(np.zeros((self.grid_height,
                                                        self.grid_width)),
                                              index=np.linspace(1,
                                                                self.grid_height,
                                                                self.grid_height,
                                                                dtype=int),
                                              columns=np.linspace(1,
                                                                  self.grid_width,
                                                                  self.grid_width,
                                                                  dtype=int))
        optimal_policy = {}
        for step in range(1, self.horizon + 1):
            optimal_policy[step] = optimal_policy_wo_hrzn.copy()
            for i in range(1, self.grid_height + 1):
                for j in range(1, self.grid_width + 1):
                    # remembering indices for q_table are x, y values
                    optimal_policy[step].loc[i, j] = \
                        self.q_table[step].columns[self.q_table[step].loc[j,
                                                                          i].argmax()]
        return optimal_policy

    def pretty_print_policy(self, policy, step):
        assert step >= 1, 'step must be at least 1'
        # arrow_dict = {0: '>', 1: 'v', 2: '<', 3: '^', 'NaN': 0}
        arrow_dict = {0: '>', 1: '<', 'NaN': 0}
        pretty_policy = pd.DataFrame(np.zeros((self.grid_height,
                                               self.grid_width)),
                                     index=np.linspace(1, self.grid_height,
                                                       self.grid_height,
                                                       dtype=int),
                                     columns=np.linspace(1, self.grid_width,
                                                         self.grid_width,
                                                         dtype=int))
        for i in range(1, self.grid_height + 1):
            for j in range(1, self.grid_width + 1):
                pretty_policy.loc[i, j] = arrow_dict[policy[step].loc[i, j]]
        print(pretty_policy)

    def visualize_agent(self, num_episodes=1):
        for episode in range(num_episodes):
            state = self.env.reset()
            for step in range(1, self.horizon + 1):
                self.env.render()
                time.sleep(0.5)

                action = self.q_table[step].columns[self.q_table[step].loc[
                    state].argmax()]
                new_state, reward, done, info = self.env.step(action)
                state = new_state

                if step == self.horizon:
                    time.sleep(2)
