import torch
import torch.distributed as dist
import time

ACTION_TAG = 0
OBS_REWARD_TAG = 1

class PolicyClient:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def run(self, num_steps=5):
        for step in range(num_steps):
            action = torch.zeros(1, device=self.device)
            dist.recv(action, src=0, tag=ACTION_TAG)
            print(f"[PolicyClient] Received action: {action.item()}")
            obs = action * 2  # Dummy observation
            reward = action * 0.1  # Dummy reward
            obs_rew = torch.tensor([obs.item(), reward.item()], device=self.device)
            print(f"[PolicyClient] Sending obs/reward: {obs_rew.tolist()}")
            dist.send(obs_rew, dst=0, tag=OBS_REWARD_TAG)
            time.sleep(0.5)
