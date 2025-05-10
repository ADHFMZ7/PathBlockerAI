import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from itertools import count
import math

from config import TrainSettings
from env import Env, Action
from agents.dqn import DQN, ReplayBuffer
from ui.render import Renderer

# Set up matplotlib for interactive plotting
plt.ion()

def select_action(state, policy_net, n_actions, config, steps_done):
    """
    Selects an action using an epsilon-greedy policy
    """
    sample = random.random()
    eps_threshold = config.eps_end + (config.eps_start - config.eps_end) * \
                    math.exp(-1. * steps_done / config.eps_decay)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # Use the policy network to select the action with highest Q-value
            return torch.argmax(policy_net(state)).item()
    else:
        # Select a random action
        return random.randrange(n_actions)


def optimize_model(policy_net, target_net, optimizer, memory, config):
    """
    Performs a single step of optimization for the policy network
    """
    if len(memory) < config.batch_size:
        return
    
    # Sample a batch from the replay memory
    states, actions, rewards, next_states, dones = memory.sample(config.batch_size)
    
    # Convert to tensors
    states = torch.stack([s for s in states])
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.stack([s for s in next_states])
    dones = torch.tensor(dones, dtype=torch.bool)  # Remove unsqueeze(1)
    
    # Calculate Q values for current states
    current_q_values = policy_net(states).gather(1, actions)
    
    # Calculate Q values for next states using target network
    with torch.no_grad():
        # Correctly handle the masking of terminal states
        next_q_values = torch.zeros(config.batch_size, 1)
        # Only calculate next Q values for non-terminal states
        non_terminal_mask = ~dones
        next_q_values[non_terminal_mask] = target_net(next_states[non_terminal_mask]).max(1, keepdim=True)[0]
    
    # Calculate the expected Q values
    expected_q_values = rewards + (config.gamma * next_q_values)
    
    # Calculate loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(current_q_values, expected_q_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # Apply gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
    optimizer.step()


def train():
    """
    Main training loop
    """
    config = TrainSettings()
    env = Env(config.game_settings)
    
    # Create the renderer if needed (set to None to disable rendering)
    renderer = Renderer(config.game_settings) if config.game_settings.render else None
    
    # Determine input and output dimensions
    initial_observation = env.reset()
    n_observations = len(initial_observation)
    n_actions = len(env.action_space)
    
    # Create policy and target networks
    policy_net = DQN(n_observations, n_actions)
    target_net = DQN(n_observations, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Set target network to evaluation mode
    
    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    
    # Initialize replay memory
    memory = ReplayBuffer(config.memory_size)
    
    # Training metrics
    episode_rewards = []
    episode_durations = []
    success_rates = []
    
    episodes_window = 100  # Window size for calculating success rate
    success_count = 0  # Count successful episodes within the window
    
    steps_done = 0
    
    # Set up the plots with a more compact size
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.4)  # Add space between subplots
    
    # Main training loop
    for episode in range(config.num_episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        
        # Episode loop
        for step in range(config.max_steps_per_episode):
            # Select and perform an action
            action_idx = select_action(state, policy_net, n_actions, config, steps_done)
            action = env.action_space[action_idx]
            next_state, reward, done = env.step(action)
            
            steps_done += 1
            total_reward += reward
            
            # Store transition in memory
            memory.add(state, action_idx, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Perform optimization step on policy network
            optimize_model(policy_net, target_net, optimizer, memory, config)
            
            # Render the environment if enabled
            if renderer:
                renderer.draw(env)
            
            # Break if episode is done
            if done:
                break
        
        # Update target network after configured number of episodes
        if episode % config.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Store episode metrics
        episode_rewards.append(total_reward)
        episode_durations.append(step + 1)  # +1 because step is 0-indexed
        
        # Track success (positive reward means success in most RL tasks)
        # You might need to adjust this condition based on your specific task
        is_success = total_reward > 0
        
        # Update success count and rate
        if episode < episodes_window:
            # For the first window, just count successes
            if is_success:
                success_count += 1
            success_rate = success_count / (episode + 1)
        else:
            # For subsequent windows, maintain a rolling window
            # Remove contribution of oldest episode and add newest
            old_episode_idx = episode - episodes_window
            old_success = episode_rewards[old_episode_idx] > 0
            if old_success:
                success_count -= 1
            if is_success:
                success_count += 1
            success_rate = success_count / episodes_window
        
        success_rates.append(success_rate)
        
        # Print progress
        print(f"Episode {episode+1}/{config.num_episodes}: Reward = {total_reward}, Steps = {step+1}, Success Rate = {success_rate:.2f}")
        
        # Update the plots every few episodes to avoid slowing down training
        if episode % 5 == 0 or episode == config.num_episodes - 1:
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Plot rewards
            ax1.plot(episode_rewards)
            ax1.set_title('Training Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            
            # Plot episode durations
            ax2.plot(episode_durations)
            ax2.set_title('Time Until Termination')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            
            # Plot success rate
            ax3.plot(success_rates)
            ax3.set_title('Success Rate')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate (moving avg)')
            ax3.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
    
    # Save the trained model
    torch.save(policy_net.state_dict(), "trained_model.pth")
    
    # Keep the plot open at the end of training
    print("Training complete! Close the plot window to exit.")
    plt.ioff()
    plt.show()
    
    return policy_net


if __name__ == "__main__":
    train()








