from collections import deque
from torch import nn
import random

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, obs):
        return self.net(obs)

class ReplayBuffer:

    def __init__(self, maxlen: int = 100_000):
        self._q = deque(maxlen=maxlen) 

    def add(self, state, action, reward, next_state, done):
        self._q.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._q, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self._q)

