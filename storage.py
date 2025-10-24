import torch
import numpy as np

class RolloutStorage:
    def __init__(self, num_steps, obs_shape, device):
        self.device = device
        # Initialize storage tensors
        self.obs = torch.zeros(num_steps + 1, *obs_shape).to(device)
        self.actions = torch.zeros(num_steps, 1, dtype=torch.long).to(device)
        self.log_probs = torch.zeros(num_steps, 1).to(device)
        self.rewards = torch.zeros(num_steps, 1).to(device)         # extrinsic reward
        self.int_rewards = torch.zeros(num_steps, 1).to(device)     # intrinsic reward
        self.values = torch.zeros(num_steps + 1, 1).to(device)
        self.dones = torch.zeros(num_steps, 1).to(device)

        self.num_steps = num_steps
        self.step = 0

    def insert(self, obs, action, log_prob, value, reward, int_reward, done):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        self.rewards[self.step] = reward      # ext_reward
        self.int_rewards[self.step] = int_reward # int_reward
        self.dones[self.step] = done
        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_value):
        self.obs[self.num_steps] = last_obs
        self.values[self.num_steps] = last_value

    def reset(self):
        self.step = 0

    def get_data(self):
        return (self.obs, self.actions, self.log_probs, self.values,
                self.rewards, self.int_rewards, self.dones)
