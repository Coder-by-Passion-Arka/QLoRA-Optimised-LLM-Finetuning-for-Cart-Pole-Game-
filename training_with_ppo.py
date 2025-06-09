import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import argparse
import os
from llm_rl_improved_agent import LLMRLImprovedAgent
import time
import warnings
import random  # Added for experience replay
warnings.filterwarnings('ignore')

def train_agent(episodes=100, max_steps=500, batch_size=32, save_dir="models"):
    """Train the agent with Unsloth-optimized PPO"""
    os.makedirs(save_dir, exist_ok=True) 
    
    # Create environments
    env = gym.make('CartPole-v1')
    env_display = gym.make('CartPole-v1', render_mode="human")
    
    # Get environment dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize Unsloth-optimized agent
    agent = LLMRLImprovedAgent(
        state_size=state_size,
        action_size=action_size,
        model_name="unsloth/llama-3.1-8b-bnb-4bit"  # Unsloth pretrained model
    )
    
    # Training metrics
    scores = []
    epsilon_values = []
    losses = {
        'policy': [],  # Policy loss
        'value': [], # Value loss
        'entropy': [], # Entropy loss
        'kl': [], # KL divergence loss
        'clipfrac': [], # Clipping fraction
        'grad_norm': [] # Gradient norm
    }
    
    # Training loop with progress bar
    progress_bar = tqdm(range(episodes), desc='Training Progress')
    for e in progress_bar:
        state, _ = env.reset()
        total_reward = 0
        episode_states, episode_actions, episode_rewards, episode_dones = [], [], [], []
        
        for step in range(max_steps):
            # Get action from agent
            action = agent.act(state)
            
            # Environment step
            next_state, reward, done, _, _ = env.step(action)
            
            # Store experience
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            
            # Store for experience replay
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update metrics
        scores.append(total_reward)
        epsilon_values.append(agent.epsilon)
        
        # Convert to numpy arrays
        states = np.array(episode_states)
        actions = np.array(episode_actions)
        rewards = np.array(episode_rewards, dtype=np.float32)
        dones = np.array(episode_dones, dtype=np.float32)
        
        # Unsloth-optimized PPO training
        loss_metrics = agent.train_ppo(states, actions, rewards) # Storing the Loss values 
        for key in losses:
            losses[key].append(loss_metrics.get(key, 0))
        
        # Experience replay for DQN
        if len(agent.memory) >= batch_size:
            minibatch = random.sample(agent.memory, batch_size)
            for s, a, r, ns, d in minibatch:
                # DQN update remains unchanged
                target = r if d else r + agent.gamma * np.amax(agent.rl_model.predict(ns.reshape(1, -1), verbose=0)[0])
                target_f = agent.rl_model.predict(s.reshape(1, -1), verbose=0)
                target_f[0][a] = target
                agent.rl_model.fit(s.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        # Update exploration rate
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        
        # Progress tracking
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        progress_bar.set_postfix({
            'Score': total_reward,
            'Avg': f"{avg_score:.1f}",
            'Epsilon': f"{agent.epsilon:.3f}"
        })
        
        # Early stopping
        if avg_score >= env.spec.reward_threshold:
            print(f"\nEnvironment solved in {e+1} episodes with average score: {avg_score:.2f}")
            # print(f"\nSolved in {e+1} episodes!")
            agent.save(os.path.join(save_dir, "cartpole_solved"))
            break
        
        # Visualizating the Model's performance every 5 episodes 
        if (e + 1) % 5 == 0:
            visualize_episode(env_display, agent, state_size, max_steps)
    
    # Save final results
    plot_results(scores, epsilon_values, losses)
    agent.save(os.path.join(save_dir, "cartpole_final"))
    env.close()
    env_display.close()
    return scores

# Rest of the code (visualize_episode, plot_results, etc.) remains unchanged...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CartPole with Unsloth-optimized LLM-RL')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)  # Increased for Unsloth
    parser.add_argument('--save-dir', type=str, default='unsloth_models')
    args = parser.parse_args()
    
    print(f"Training with Unsloth optimization for {args.episodes} episodes")
    train_agent(
        episodes=args.episodes,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
def visualize_episode(env, agent, state_size, max_steps):
    """Visualize a single episode"""
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):
        env.render()
        action = agent.act(state, training=False)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    print(f"Episode finished with total reward: {total_reward}")
    env.close()
    
def plot_results(scores, epsilon_values, losses):
    """Plot training results"""
    plt.figure(figsize=(12, 8))
    
    # Plot scores
    plt.subplot(2, 2, 1)
    plt.plot(scores, label='Scores', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.legend()
    
    # Plot epsilon values
    plt.subplot(2, 2, 2)
    plt.plot(epsilon_values, label='Epsilon', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.legend()
    
    # Plot losses
    plt.subplot(2, 2, 3)
    for key in losses:
        if losses[key]:
            plt.plot(losses[key], label=key)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Loss Metrics')
    plt.legend()
    
    # Show plots
    plt.tight_layout()
    plt.show()
    plt.close()
    # Save plots
    plt.savefig(os.path.join("plots", "training_results.png"))
    os.makedirs("plots", exist_ok=True)