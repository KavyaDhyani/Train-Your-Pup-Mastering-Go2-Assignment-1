import gym
import numpy as np
import matplotlib.pyplot as plt
from cart_pole_def import CartPoleQAgent  # Assuming the class is in a file named `cart_pole_def.py`

# Initialize the CartPole environment
env = gym.make("CartPole-v1")
state, _ = env.reset()

# Define bounds for the state space
upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low
upper_bounds[1] = 3.0  # Velocity max
upper_bounds[3] = 10.0  # Angular velocity max
lower_bounds[1] = -3.0  # Velocity min
lower_bounds[3] = -10.0  # Angular velocity min

# Define hyperparameters
number_of_bins = [30, 30, 30, 30]
alpha = 0.1
gamma = 1.0
epsilon = 0.5
number_of_episodes = 30000

# Initialize the Q-learning agent
agent = CartPoleQAgent(
    env=env,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon,
    num_episodes=number_of_episodes,
    num_bins=number_of_bins,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
)
'''
# Train the agent
agent.simulate()

# Save the Q-table
agent.save_q_table(filename="q_table.npy")

'''
# Evaluate the learned policy
total_rewards = []
for i in range(10):  # Run 10 simulations with the learned policy
    rewards, env_with_render = agent.simulate_with_learned_policy(q_table_file="q_table.npy")
    total_rewards.append(np.sum(rewards))
    env_with_render.close()
'''
# Plot the convergence of rewards
plt.figure(figsize=(12, 5))
plt.plot(agent.reward_per_episode, color="blue", linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.yscale("log")
plt.title("Convergence of Rewards Over Episodes")
plt.savefig("cart_pole.png")
plt.show()

'''
# Print the sum of rewards for the optimal policy
print(f"Total rewards across 10 simulations: {np.mean(total_rewards):.2f}")

