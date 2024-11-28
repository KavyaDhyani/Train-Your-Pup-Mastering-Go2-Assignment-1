import gym
import numpy as np
import time

class CartPoleQAgent:
    def __init__(self, env, alpha, gamma, epsilon, num_episodes, num_bins, lower_bounds, upper_bounds, epsilon_decay=0.99, min_epsilon=0.05):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_bins = num_bins
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_space = env.action_space.n
        self.reward_per_episode = []
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        return np.random.uniform(low=0, high=1, size=(
            self.num_bins[0], self.num_bins[1], self.num_bins[2], self.num_bins[3], self.action_space))

    def get_state_indices(self, state):
        position, velocity, angle, angular_velocity = state
        position_bins = np.linspace(-2.4, 2.4, self.num_bins[0])
        velocity_bins = np.linspace(self.lower_bounds[1], self.upper_bounds[1], self.num_bins[1])
        angle_bins = np.linspace(-0.21, 0.21, self.num_bins[2])
        angular_velocity_bins = np.linspace(self.lower_bounds[3], self.upper_bounds[3], self.num_bins[3])
        
        return (
            self.get_bin_index(position, position_bins),
            self.get_bin_index(velocity, velocity_bins),
            self.get_bin_index(angle, angle_bins),
            self.get_bin_index(angular_velocity, angular_velocity_bins)
        )

    def get_bin_index(self, value, bins):
        return np.maximum(np.digitize(value, bins) - 1, 0)

    def choose_action(self, state, episode_idx):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)  
        return self.get_greedy_action(state)  

    def get_greedy_action(self, state):
        state_indices = self.get_state_indices(state)
        return np.random.choice(np.where(self.q_table[state_indices] == np.max(self.q_table[state_indices]))[0])

    def update_q_table(self, state, action, reward, next_state, terminal):
        state_indices = self.get_state_indices(state)
        next_state_indices = self.get_state_indices(next_state)
        
        q_max_next = np.max(self.q_table[next_state_indices]) if not terminal else 0
        td_error = reward + self.gamma * q_max_next - self.q_table[state_indices + (action,)]
        self.q_table[state_indices + (action,)] += self.alpha * td_error

    def simulate(self):
        for episode_idx in range(self.num_episodes):
            total_rewards = []
            state, _ = self.env.reset()
            state = list(state)
            terminal = False
            
            while not terminal:
                action = self.choose_action(state, episode_idx)
                next_state, reward, terminal, _, _ = self.env.step(action)
                
                if self.is_terminal_state(next_state):
                    reward -= 10
                    terminal = True
                
                total_rewards.append(reward)
                self.update_q_table(state, action, reward, next_state, terminal)
                state = list(next_state)

            self.reward_per_episode.append(np.sum(total_rewards))
            print(f"Episode {episode_idx}: Total reward = {np.sum(total_rewards)}")
            
            # Decay epsilon
            if episode_idx >= 7000:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def is_terminal_state(self, state):
        position, _, angle, _ = state
        return position <= -2.4 or position >= 2.4 or angle <= -0.21 or angle >= 0.21

    def simulate_with_learned_policy(self, q_table_file="q_table.npy"):
        self.load_q_table(q_table_file)
        env1 = gym.make('CartPole-v1', render_mode='human')
        state, _ = env1.reset()
        env1.render()
        rewards = []
        
        for _ in range(1000):
            action = self.get_greedy_action(state)
            state, reward, terminated, truncated, info = env1.step(action)
            rewards.append(reward)
            time.sleep(0.05)
            if terminated:
                break
        
        return rewards, env1

    def save_q_table(self, filename="q_table.npy"):
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.npy"):
        self.q_table = np.load(filename)
        print(f"Q-table loaded from {filename}")
