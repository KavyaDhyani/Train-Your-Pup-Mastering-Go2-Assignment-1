import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def initialize_environment(render_mode=False):
    env = gym.make("MountainCar-v0", render_mode="human" if render_mode else None)
    return env


def create_state_spaces(env, num_bins=20):
    position_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_bins)
    velocity_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num_bins)
    return position_space, velocity_space


def load_q_table(is_training, filename='mountain_car.pkl', position_space=None, velocity_space=None, action_space=None):
    if is_training:
        return np.zeros((len(position_space), len(velocity_space), action_space.n))
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def update_q_value(q_table, state, action, reward, next_state, learning_rate, discount_factor):
    pos_idx, vel_idx = state
    next_pos_idx, next_vel_idx = next_state
    best_next_action = np.max(q_table[next_pos_idx, next_vel_idx, :])
    current_q_value = q_table[pos_idx, vel_idx, action]

    updated_q_value = current_q_value + learning_rate * (reward + discount_factor * best_next_action - current_q_value)
    q_table[pos_idx, vel_idx, action] = updated_q_value


def discretize_state(state, position_space, velocity_space):
    pos_idx = np.digitize(state[0], position_space)
    vel_idx = np.digitize(state[1], velocity_space)
    return pos_idx, vel_idx


def train(env, episodes, position_space, velocity_space, epsilon_decay_rate, learning_rate, discount_factor):
    q_table = np.zeros((len(position_space), len(velocity_space), env.action_space.n))
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    epsilon = 1.0

    for episode in range(episodes):
        state = env.reset()[0]
        state_idx = discretize_state(state, position_space, velocity_space)
        terminated = False
        total_rewards = 0

        while not terminated and total_rewards > -1000:
            action = np.random.choice(env.action_space.n) if rng.random() < epsilon else np.argmax(q_table[state_idx])
            next_state, reward, terminated, _, _ = env.step(action)

            next_state_idx = discretize_state(next_state, position_space, velocity_space)
            update_q_value(q_table, state_idx, action, reward, next_state_idx, learning_rate, discount_factor)

            state_idx = next_state_idx
            total_rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[episode] = total_rewards

    return q_table, rewards_per_episode


def evaluate(env, q_table, position_space, velocity_space, episodes):
    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()[0]
        state_idx = discretize_state(state, position_space, velocity_space)
        terminated = False
        total_rewards = 0

        while not terminated:
            action = np.argmax(q_table[state_idx])
            next_state, reward, terminated, _, _ = env.step(action)
            state_idx = discretize_state(next_state, position_space, velocity_space)
            total_rewards += reward

        rewards_per_episode[episode] = total_rewards

    return rewards_per_episode


def plot_rewards(rewards_per_episode, filename='mountain_car.png'):
    mean_rewards = np.zeros_like(rewards_per_episode)
    for i in range(len(rewards_per_episode)):
        mean_rewards[i] = np.mean(rewards_per_episode[max(0, i - 100):i + 1])

    plt.plot(mean_rewards)
    plt.savefig(filename)


def save_q_table(q_table, filename='mountain_car.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)


def main(episodes=1000, is_training=True, render=False):
    env = initialize_environment(render)
    position_space, velocity_space = create_state_spaces(env)
    q_table = load_q_table(is_training, position_space=position_space, velocity_space=velocity_space, action_space=env.action_space)

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon_decay_rate = 2 / episodes

    if is_training:
        q_table, rewards_per_episode = train(env, episodes, position_space, velocity_space, epsilon_decay_rate, learning_rate, discount_factor)
        save_q_table(q_table)
    else:
        rewards_per_episode = evaluate(env, q_table, position_space, velocity_space, episodes)

    plot_rewards(rewards_per_episode)


if __name__ == '__main__':
    main(episodes=5, is_training=False, render=True)
