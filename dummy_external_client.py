import os
import pickle
import time

import gymnasium as gym
import numpy as np
import torch.distributed as dist

from ray.rllib.core import (
    Columns,
    COMPONENT_RL_MODULE,
)
from ray.rllib.env.external.rllink import (
    get_rllink_message,
    send_rllink_message,
    RLlink,
)
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import softmax

torch, _ = try_import_torch()

# Initialize NCCL process group
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

def _dummy_external_client(port: int = 5556):
    """A dummy client that runs CartPole and acts as a testing external env."""

    initialized = False

    print("Starting sim")
    def _set_state(msg_body, rl_module):
        rl_module.set_state(msg_body[COMPONENT_RL_MODULE])

    def _init():
        nonlocal initialized
        if not initialized:
            # dist.init_process_group(backend="nccl", rank=0, world_size=2)
            dist.init_process_group(backend="gloo", rank=0, world_size=2)
            initialized = True

    print("Creating process group with NCCL backend")
    # dist.init_process_group(backend="nccl", rank=0, world_size=2)
    
    # Initialize environment
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    print("obs", obs)
    episode = SingleAgentEpisode(observations=[obs])
    episodes = [episode]

    env_steps_per_sample = 10  # Example value

    # Initialize RLModule (dummy initialization for NCCL example)
    rl_module = None  # Replace with actual RLModule initialization logic

    while True:
        print("Stepping sim")
        # Perform action inference using the RLModule.
        logits = torch.tensor([0.5, 0.5])  # Dummy logits for example
        action_probs = softmax(logits.numpy())
        action = int(np.random.choice(list(range(env.action_space.n)), p=action_probs))
        logp = float(np.log(action_probs[action]))

        # Perform the env step.
        obs, reward, terminated, truncated, _ = env.step(action)

        # Collect step data.
        episode.add_env_step(
            action=action,
            reward=reward,
            observation=obs,
            terminated=terminated,
            truncated=truncated,
            extra_model_outputs={
                Columns.ACTION_DIST_INPUTS: logits.numpy(),
                Columns.ACTION_LOGP: logp,
            },
        )

        # Initialize tensor before moving to GPU
        tensor = torch.zeros(10)  # Example initialization, adjust as needed
        tensor = tensor.cuda()

        _init()

        # Send data to server using NCCL
        # dist.send(tensor, dst=1)

        # Prepare to receive data on the GPU
        recv_tensor = torch.empty_like(tensor, device='cuda')
        # dist.recv(recv_tensor, src=1)

        # Move received tensor back to CPU for processing
        # updated_state = pickle.loads(bytes(recv_tensor.cpu().tolist()))
        # updated_state = pickle.loads(bytes(recv_tensor.cpu().numpy().astype('uint8')))

        # _set_state(updated_state, rl_module)

        episodes = []
        if not episode.is_done:
            episode = episode.cut()
            episodes.append(episode)

        # If episode is done, reset env and create a new episode.
        if episode.is_done:
            obs, _ = env.reset()
            episode = SingleAgentEpisode(observations=[obs])
            episodes.append(episode)

        time.sleep(2)
