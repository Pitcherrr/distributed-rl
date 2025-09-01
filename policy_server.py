import torch
import torch.distributed as dist
import time

ACTION_TAG = 0
OBS_REWARD_TAG = 1

class PolicyServer:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def serve(self, num_steps=5):
        for step in range(num_steps):
            action = torch.tensor([step], dtype=torch.float32, device=self.device)
            print(f"[PolicyServer] Sending action: {action.item()}")
            dist.send(action, dst=1, tag=ACTION_TAG)
            obs_rew = torch.zeros(2, device=self.device)
            dist.recv(obs_rew, src=1, tag=OBS_REWARD_TAG)
            print(f"[PolicyServer] Received obs/reward: {obs_rew.tolist()}")
            time.sleep(0.5)
