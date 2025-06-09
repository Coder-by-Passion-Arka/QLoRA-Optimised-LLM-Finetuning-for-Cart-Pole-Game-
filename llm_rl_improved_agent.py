# This file defines an improved LLM-based RL agent using Unsloth optimizations.
import os
import torch
import random  # For experience replay
import warnings
import numpy as np
import gymnasium as gym
from collections import deque
import torch.nn.functional as F
from unsloth import FastLanguageModel
from trl import PPOTrainer, PPOConfig
from unsloth import FastLanguageModel
from trl import PPOTrainer, PPOConfig
from transformers import TrainingArguments
# from unsloth import AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLMWithValueHead
warnings.filterwarnings('ignore')

class LLMRLImprovedAgent:
    def __init__(self, state_size, action_size, model_name="unsloth/llama-3.1-8b-bnb-4bit"):
        # Initialize device and environment parameters
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("==================================================================")
        print(f"\nUsing device: {self.device}\n")

        # 1. Initialize Unsloth-optimized model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=4096, # 8192 , 2048
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32, 
            load_in_4bit=True, # Use 4bit quantization to reduce memory usage.
            token=os.environ["HUGGING_FACE_API"], # Hugging Face API token
        )

        # 2. Configure LoRA with Unsloth optimizations
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=32,  # Higher rank for better policy learning
            lora_alpha=64, # LoRA scaling factor 
            target_modules=[
                "q_proj", # q_proj = Projection of Query values to contextualized embeddings
                "k_proj", # k_proj = Projection of Key values to contextualized embeddings
                "v_proj", # v_proj = Projection of Value values to contextualized embeddings
                "o_proj" # o_proj = Projection of Output values to contextualized embeddings
            ],
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        # 3. Add value head for PPO
        self.model = AutoModelForCausalLMWithValueHead(self.model)

        # 4. Initialize PPO trainer
        self.ppo_config = PPOConfig(
            batch_size=32,
            mini_batch_size=8,
            ppo_epochs=4,
            learning_rate=1e-5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            target_kl=0.01,
            seed=3407,
        )

        # 5. Training state tracking
        self.memory = deque(maxlen=2000)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.ppo_config.learning_rate,
            fused=True,  # Unsloth optimization
        )

    def train_ppo(self, states, actions, rewards):
        """Optimized PPO training with Unsloth"""
        # Convert states to prompt format
        prompts = [self._format_state_prompt(s) for s in states]
        
        # Tokenize inputs with Unsloth-optimized padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        ).to(self.device)

        # Get initial policy predictions
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            old_log_probs = F.log_softmax(outputs.logits[:, -1, :self.action_size], dim=-1)
            old_values = outputs.value[:, -1].detach()

        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            model=self.model,
            config=self.ppo_config,
            optimizer=self.optimizer,
            tokenizer=self.tokenizer,
        )

        # Training loop
        for epoch in range(self.ppo_config.ppo_epochs):
            # Get current policy predictions
            logits, values, _ = self.model(**inputs)
            new_log_probs = F.log_softmax(logits[:, -1, :self.action_size], dim=-1)
            
            # Calculate advantages
            advantages, returns = self._compute_advantages(
                rewards, 
                values[:, -1].cpu().numpy(),
                old_values.cpu().numpy(),
            )

            # PPO loss calculation
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_config.clip_range, 1.0 + self.ppo_config.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values[:, -1], returns.to(self.device))
            
            # Entropy bonus
            entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(-1).mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            # Unsloth-optimized backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def _compute_advantages(self, rewards, values, old_values):
        """Unsloth-optimized advantage calculation"""
        # Convert to numpy arrays for efficient computation
        rewards = np.array(rewards)
        values = np.array(values)
        old_values = np.array(old_values)
        
        # Generalized Advantage Estimation (GAE)
        deltas = rewards + self.ppo_config.gamma * old_values[1:] - old_values[:-1]
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards)-1)):
            advantages[t] = last_advantage = (
                deltas[t] + 
                self.ppo_config.gamma * self.ppo_config.gae_lambda * last_advantage
            )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values[:-1]
        
        return (
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
        )

    # Rest of the class remains with Unsloth optimizations...
    def _format_state_prompt(self, state):
        """Format state for LLM input"""
        return ( 
        "You are an intelligent player of the CartPole environment of OpenAI Gym. Your task is to be a master at balancing the pole on a moving cart.\n"
        "- The current state is provided as a list of four values: [cart_position, cart_velocity, pole_angle, pole_angular_velocity].\n"
        " - The environment provides you with the current state of the cart and pole, which includes:\n"
        " - Cart position: The horizontal position of the cart.\n"
        " - Cart velocity: The horizontal velocity of the cart.\n"
        " - Pole angle: The angle of the pole in radians.\n"
        " - Pole angular velocity: The angular velocity of the pole in radians.\n"
        " - You can take one of two actions:\n"
        " - 0: Push the cart to the left.\n"
        " - 1: Push the cart to the right.\n"
        " - The goal is to keep the pole balanced by moving the cart left or right.\n" 
        "Given the current state, select the best action (0 or 1) to keep the pole balanced.\n"
        "- Hint: The pole is balanced when its angle is close to zero and the cart is centered. Also remeber the laws of Physics when trying to balance the pole. If the pole is falling to the left then push the cart to the left so that relative velocity is decresed and the pole gets balanced.\n"
        "- Don't shy away from experimenting with the actions and try to balance the pole as long as possible.\n"
        f"State: {state}\n" # Current state of the environment
        "Action:" # Action reported by the LLM (0 or 1)"
        ) 
    
    def act(self, state):
        """Get action from the model based on the current state"""
        prompt = self._format_state_prompt(state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :self.action_size]
            action_probs = F.softmax(logits, dim=-1)
        
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay"""
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, path):
        """Save the model to the specified path"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    def load_model(self, path):
        """Load the model from the specified path"""
        self.model = FastLanguageModel.from_pretrained(path, device=self.device)
        self.tokenizer = FastLanguageModel.get_tokenizer(path)
        self.model.to(self.device)
        self.model.eval()

