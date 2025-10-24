import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
from model import ActorCritic # Import our network
from storage import RolloutStorage

class PPOAgent:
    def __init__(self, 
                 num_actions=18, 
                 learning_rate=2.5e-4, 
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.1,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 ppo_epochs=4,
                 num_mini_batches=4,
                 int_reward_coef=1.0):
 
        # Setup device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.int_reward_coef = int_reward_coef

        # Initialize Actor-Critic network
        self.model = ActorCritic(num_actions).to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)

    def get_action_and_value(self, state):
        # Add a batch dimension
        if state.dim() == 3:
             state = state.unsqueeze(0) 

        with torch.no_grad():
            actor_logits, critic_value = self.model(state)

        dist = distributions.Categorical(logits=actor_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, critic_value

    def train(self, storage: RolloutStorage):
        # 1. Get the collected data
        (obs, actions, old_log_probs, values, 
         ext_rewards, int_rewards, dones) = storage.get_data()

        # 2. Normalize the intrinsic rewards from this rollout
        int_rewards = (int_rewards - int_rewards.mean()) / (int_rewards.std() + 1e-8)

        # 3. Combine extrinsic and intrinsic rewards
        rewards = ext_rewards + self.int_reward_coef * int_rewards

        # 4. Calculate Advantages (GAE) and Returns (using combined rewards)
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)

        gae_advantage = 0
        for t in reversed(range(storage.num_steps)):
            # TD Error (delta)
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]

            # GAE
            gae_advantage = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae_advantage

            advantages[t] = gae_advantage
            returns[t] = gae_advantage + values[t] # Return = Advantage + Value

        # 5. Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 6. PPO Update Loop
        self.model.train() 

        # It will use `advantages` and `returns` computed from the combined reward.
        batch_size = storage.num_steps
        mini_batch_size = batch_size // self.num_mini_batches
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size).to(self.device)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                actor_logits, critic_values = self.model(mb_obs)
                dist = distributions.Categorical(logits=actor_logits)
                new_log_probs = dist.log_prob(mb_actions.squeeze(-1)).unsqueeze(-1)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() 
                critic_loss = F.mse_loss(critic_values, mb_returns)
                loss = (actor_loss 
                        + self.value_loss_coef * critic_loss 
                        - self.entropy_coef * entropy)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

