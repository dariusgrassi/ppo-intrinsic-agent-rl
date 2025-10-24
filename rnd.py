import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RNDNetwork(nn.Module):
    def __init__(self, output_features=512):
        super(RNDNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3136, out_features=output_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class RNDModule:
    def __init__(self, device, learning_rate=1e-4, output_features=512):
        self.device = device
        # Target Network (Frozen)
        self.target_net = RNDNetwork(output_features).to(device)
        self.target_net.eval() # Set to evaluation mode
        # Freeze target network parameters
        for param in self.target_net.parameters():
            param.requires_grad = False
        # Predictor Network (Trained)
        self.predictor_net = RNDNetwork(output_features).to(device)
        self.predictor_net.train() # Set to training mode

        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=learning_rate)

    def compute_intrinsic_reward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            target_features = self.target_net(obs / 255.0) # Normalize pixels
        predictor_features = self.predictor_net(obs / 255.0)
        # Calculate MSE loss (this is our reward)
        loss = F.mse_loss(predictor_features, target_features, reduction='mean')
        return loss.item()

    def train(self, obs_batch):
        obs_batch = obs_batch.to(self.device)
        with torch.no_grad():
            target_features = self.target_net(obs_batch) # Already normalized
        predictor_features = self.predictor_net(obs_batch)
        # Calculate MSE loss
        loss = F.mse_loss(predictor_features, target_features)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

