import torch
import torch.distributed as dist
import time

ACTION_TAG = 0
OBS_REWARD_TAG = 1
END_EPISODE = 2

class PolicyServer:
    def __init__(self, rank, world_size, device, backend):
        dist.init_process_group(backend)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.observation_space = None
        self.action_space = None

        self.single_tensor = torch.tensor([1], dtype=torch.float32, device=self.device)

        print("created policy server")

    def get_action(self, obs):
        obs.to(self.device)
        # send our abservation back to the network for inference 
        dist.send(obs, dst=1, tag=OBS_REWARD_TAG)
        # gen the action that comes back from the network
        action = torch.zeros(1, device=self.device)
        dist.recv(action, src=1, dst=1, tag=OBS_REWARD_TAG)
        return action

    def end_episode(self):
        dist.send(self.single_tensor, tag=END_EPISODE)


    def serve(self, num_steps=5):
        for step in range(num_steps):
            action = torch.tensor([step], dtype=torch.float32, device=self.device)
            print(f"[PolicyServer] Sending action: {action.item()}")
            dist.send(action, dst=1, tag=ACTION_TAG)
            obs_rew = torch.zeros(2, device=self.device)
            dist.recv(obs_rew, src=1, tag=OBS_REWARD_TAG)
            print(f"[PolicyServer] Received obs/reward: {obs_rew.tolist()}")
            time.sleep(0.5)
