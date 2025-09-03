import threading

import gymnasium as gym
import numpy as np

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.external.env_runner_server_for_external_inference import (
    EnvRunnerServerForExternalInference,
)
from ray.rllib.examples.envs.classes.utils.dummy_external_client import (
    _dummy_external_client,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls




if __name__ == "__main__":
 
    parser = add_rllib_example_script_args(
        default_reward=450.0, default_iters=200, default_timesteps=2000000
    )
    
    args = parser.parse_args()

    # Start the dummy CartPole "simulation".
    if args.use_dummy_client:
        threading.Thread(
            target=_dummy_external_client,
            args=(
                # Connect to the first remote EnvRunner, of - if there is no remote one -
                # to the local EnvRunner.
                args.port
                + (args.num_env_runners if args.num_env_runners is not None else 1),
            ),
        ).start()

    # Define the RLlib (server) config.
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            observation_space=gym.spaces.Box(
                float("-inf"), float("-inf"), (4,), np.float32
            ),
            action_space=gym.spaces.Discrete(2),
            # EnvRunners listen on `port` + their worker index.
            env_config={"port": args.port},
        )
        .env_runners(
            # Point RLlib to the custom EnvRunner to be used here.
            env_runner_cls=EnvRunnerServerForExternalInference,
        )
        .training(
            num_epochs=10,
            vf_loss_coeff=0.01,
        )
        .rl_module(model_config=DefaultModelConfig(vf_share_layers=True))
    )

    run_rllib_example_script_experiment(base_config, args)