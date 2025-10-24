import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_actions=18):
        super(ActorCritic, self).__init__()

        # Shared CNN Base
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Shared Linear Base
        self.flatten = nn.Flatten()
        # 64 * 7 * 7 = 3136
        self.fc_base = nn.Linear(in_features=3136, out_features=512) 

        # Two Heads
        self.actor_head = nn.Linear(in_features=512, out_features=num_actions)
        self.critic_head = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        # Normalize pixel values
        x = x / 255.0 

        # Pass through CNN Base
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and pass through shared FC layer
        x = self.flatten(x)
        x = F.relu(self.fc_base(x))

        # Get Actor and Critic outputs
        actor_logits = self.actor_head(x)
        critic_value = self.critic_head(x)

        return actor_logits, critic_value

