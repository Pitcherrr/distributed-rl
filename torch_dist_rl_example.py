
import torch
import torch.distributed as dist
import os
import time
import argparse

# Minimal RL agent and environment communication using torch.distributed
# Run this script with two processes, e.g. using torchrun or python -m torch.distributed.run

# Message tags
ACTION_TAG = 0
OBS_REWARD_TAG = 1


# Import the new abstractions
from policy_server import PolicyServer
from policy_client import PolicyClient


def main():
    parser = argparse.ArgumentParser(description="Minimal torch.distributed RL agent-env example")
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl', 'mpi'],
                        help='torch.distributed backend to use (default: gloo)')
    args = parser.parse_args()

    print(f"NCCL: {torch.distributed.is_nccl_available()}")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(args.backend)

    if args.backend == 'nccl':
        # device = torch.device(f'cuda:{rank}')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using {device}")

    if rank == 0:
        server = PolicyServer(rank, world_size, device)
        server.serve(num_steps=5)
    elif rank == 1:
        client = PolicyClient(rank, world_size, device)
        client.run(num_steps=5)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
