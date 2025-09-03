import os
import torch
import gymnasium as gym
from policy_server import PolicyServer

print("test env outside")


if __name__ == "__main__":

    print("test env")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    rl_server = PolicyServer(rank, world_size, torch.device("cuda"), "nccl")
    
    # Initialise the environment
    env = gym.make("LunarLander-v3")

    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        # this is where you would insert your policy
        # action = env.action_space.sample()

        print(f"observation {observation}")

        action = rl_server.get_action(torch.tensor(observation))

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    