import gym
import numpy as np
import matplotlib.pyplot as plt
from cart_pole_def import CartPoleQAgent  


env = gym.make("CartPole-v1")
state, _ = env.reset()


upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low
upper_bounds[1] = 3.0  
upper_bounds[3] = 10.0  
lower_bounds[1] = -3.0  
lower_bounds[3] = -10.0  


number_of_bins = [30, 30, 30, 30]
alpha = 0.1
gamma = 1.0
epsilon = 0.5
number_of_episodes = 30000


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

total_rewards = []
for i in range(10):  
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

print(f"Total rewards across 10 simulations: {np.mean(total_rewards):.2f}")

