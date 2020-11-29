import pandas as pd
import numpy as np
import random
import time
from IPython.display import clear_output


class EGreedyQLearner():
    def __init__(self, env, init_val=0):
        self.env = env
        self.init_val = init_val
        self.num_actions = env.action_space.n
        self.grid_height = env.height - 2  # grid has padding of 1
        self.grid_width = env.width - 2  # grid has padding of 1
        self.q_table = self.init_q_table()

    def init_q_table(self):
        states = []
        for a in range(1, self.grid_height + 1):
            for b in range(1, self.grid_width + 1):
                states.append((b, a))
        init_vals = np.random.rand(len(states), self.num_actions) + \
                    self.init_val

        q_table = pd.DataFrame(init_vals,
                               index=pd.MultiIndex.from_tuples(states),
                               columns=np.linspace(
                                   0, self.num_actions - 1, self.num_actions,
                                   dtype=int))
        return q_table

    def q_learning(self, num_episodes, learning_rate=0.1, discount_rate=0.9,
                   exploration_rate=0.1, horizon=10, ep_chunk=1,
                   print_rew=False):

        rewards = np.zeros(num_episodes)
        steps = np.zeros(num_episodes)

        for episode in range(1, num_episodes + 1):
            # print(f'we are on episode: {episode}')
            state = self.env.reset()
            rewards_current_episode = 0
            step = 1
            while True:
                # Exploration and Exploitation
                exploration_rate_threshold = random.uniform(0, 1)
                if exploration_rate_threshold > exploration_rate:
                    action = self.q_table.columns[self.q_table.loc[
                        state].argmax()]
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, info = self.env.step(action)

                # update Q-table
                td_error = reward + discount_rate * self.q_table.loc[
                    new_state].max() - \
                           self.q_table.loc[state, action]
                self.q_table.loc[state, action] = self.q_table.loc[state,
                                                                  action] + \
                                             learning_rate * (td_error)

                state = new_state
                rewards_current_episode += reward

                if done == True or step == max_steps_per_episode:
                    steps[episode - 1] = step
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
        return steps, rewards

    def get_optimal_policy(self):
        optimal_policy = pd.DataFrame(np.zeros((self.grid_height,
                                                self.grid_width)),
                                      index=np.linspace(1, self.grid_height,
                                                        self.grid_height,
                                                        dtype=int),
                                      columns=np.linspace(1, self.grid_width,
                                                          self.grid_width,
                                                          dtype=int))
        for i in range(1, self.grid_height + 1):
            for j in range(1, self.grid_width + 1):
                # remembering indices for q_table are x, y values
                optimal_policy.loc[i, j] = \
                    self.q_table.columns[self.q_table.loc[j, i].argmax()]
        return optimal_policy

    def pretty_print_policy(self, policy):
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
                pretty_policy.loc[i, j] = arrow_dict[policy.loc[i, j]]
        print(pretty_policy)

    def visualize_agent(self, num_episodes=1, max_steps_per_episode=100):
        for episode in range(num_episodes):
            state = self.env.reset()
            for step in range(max_steps_per_episode):
                time.sleep(0.2)
                self.env.render()

                action = self.q_table.columns[self.q_table.loc[state].argmax()]
                new_state, reward, done, info = self.env.step(action)
                state = new_state

                if done:
                    self.env.render()
                    break

    def set_q_table_vals(self, state_list, val=0):
        for state in state_list:
            self.q_table.loc[state] = val