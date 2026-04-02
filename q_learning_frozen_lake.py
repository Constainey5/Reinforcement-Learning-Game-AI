
import numpy as np
import gymnasium as gym
import random
import time

# --- Configuration ---
ENVIRONMENT_NAME = "FrozenLake-v1"
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

# Epsilon-greedy parameters
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.001

print(f"
--- Reinforcement Learning: Q-Learning for {ENVIRONMENT_NAME} ---")

# --- Environment Setup ---
env = gym.make(ENVIRONMENT_NAME, is_slippery=False, render_mode=None) # Set render_mode to None for faster training

# Initialize Q-table with zeros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

print(f"Q-table initialized with shape: {q_table.shape}")

# --- Q-Learning Algorithm ---
def train_q_learning_agent():
    epsilon = EPSILON_START
    rewards_per_episode = []

    for episode in range(NUM_EPISODES):
        state = env.reset()[0]  # env.reset() returns (observation, info)
        done = False
        rewards_current_episode = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            # Epsilon-greedy strategy: explore or exploit
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state, :])  # Exploit learned values

            # Take action and observe new state and reward
            new_state, reward, done, truncated, info = env.step(action)

            # Update Q-table
            q_table[state, action] = q_table[state, action] * (1 - LEARNING_RATE) +                                      LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done or truncated:
                break
        
        # Decay epsilon
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-EPSILON_DECAY_RATE * episode)
        rewards_per_episode.append(rewards_current_episode)

    print("
--- Training complete ---")
    print(f"Average reward over last 100 episodes: {sum(rewards_per_episode[-100:]) / 100}")
    return q_table

# --- Evaluation ---
def evaluate_agent(q_table_trained):
    print("
--- Evaluating trained agent ---")
    total_rewards = 0
    num_eval_episodes = 100
    
    # Create a new environment for rendering during evaluation
    eval_env = gym.make(ENVIRONMENT_NAME, is_slippery=False, render_mode='human')

    for episode in range(num_eval_episodes):
        state = eval_env.reset()[0]
        done = False
        rewards_current_episode = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = np.argmax(q_table_trained[state, :]) # Always exploit during evaluation
            new_state, reward, done, truncated, info = eval_env.step(action)
            rewards_current_episode += reward
            state = new_state
            if done or truncated:
                break
        total_rewards += rewards_current_episode
        time.sleep(0.1) # Small delay for human observation

    eval_env.close()
    print(f"Average reward over {num_eval_episodes} evaluation episodes: {total_rewards / num_eval_episodes}")


if __name__ == "__main__":
    trained_q_table = train_q_learning_agent()
    # To visualize the agent, uncomment the line below and run in an environment with display capabilities
    # evaluate_agent(trained_q_table)
    print("Q-Learning agent training script finished.")
