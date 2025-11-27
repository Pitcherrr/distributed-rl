from collections import defaultdict
import os
import pickle
import socket
import threading
import time
from typing import Collection, DefaultDict, List, Optional, Union

import torch.distributed as dist

from ray.rllib.core import (
    COMPONENT_RL_MODULE,
    DEFAULT_AGENT_ID,
    DEFAULT_MODULE_ID,
)
from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.external.rllink import (
    get_rllink_message,
    send_rllink_message,
    RLlink,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_LEN_MAX,
    EPISODE_LEN_MEAN,
    EPISODE_LEN_MIN,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    WEIGHTS_SEQ_NO,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeID, StateDict
from ray.util.annotations import DeveloperAPI

torch, _ = try_import_torch()


@DeveloperAPI
class EnvRunnerServerForNCCL(EnvRunner, Checkpointable):
    def __init__(self, *, config, backend='nccl', **kwargs):
        super().__init__(config=config, **kwargs)

        self.backend = backend

        # Dynamically select backend based on GPU availability
        # if not torch.cuda.is_available():
            # raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")

        self.rank = 1
        self.world_size = 2
        self.master_addr = '127.0.0.1'
        self.master_port = 29500
        self.client_socket = None

        print(f"Starting nccl env runner {self.worker_index}")

        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)

        # Validate rank and world size
        # if not (0 <= self.rank < self.world_size):
            # raise ValueError(f"Invalid rank {self.rank} for world size {self.world_size}. Rank must be in range [0, {self.world_size - 1}].")

        print(f"[EnvRunnerServerForNCCL] Initializing process group with backend={self.backend}, rank={self.rank}, world_size={self.world_size}")
        # Initialize the process group
        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.world_size)

        self._blocked_on_state = False
        self._sample_lock = threading.Lock()

        # Start a background thread for RLlink communication
        self.thread = threading.Thread(target=self._client_message_listener, daemon=True)
        self.thread.start()

    def send_action(self, action, dst):
        print(f"[EnvRunnerServerForNCCL] Sending action {action.tolist()} to rank {dst}")
        dist.send(action, dst=dst)

    def receive_obs_reward(self, obs_rew, src):
        print(f"[EnvRunnerServerForNCCL] Receiving obs/reward from rank {src}")
        dist.recv(obs_rew, src=src)

    def _client_message_listener(self):
        while True:
            try:
                msg_type, msg_body = get_rllink_message(self.client_socket)

                if msg_type == RLlink.PING:
                    self._send_pong_message()

                elif msg_type == RLlink.GET_STATE:
                    self._send_set_state_message()

                elif msg_type == RLlink.SET_STATE:
                    self.set_state(pickle.loads(msg_body))

                elif msg_type == RLlink.EPISODES_AND_GET_STATE:
                    self._blocked_on_state = True

            except ConnectionError as e:
                print(f"[EnvRunnerServerForNCCL] Connection error: {e}")
                self._recycle_sockets()

    @override(EnvRunner)
    def sample(self, **kwargs):
        # Example of using NCCL send/recv
        action = torch.tensor([1.0], dtype=torch.float32)
        self.send_action(action, dst=1)

        obs_rew = torch.zeros(2, dtype=torch.float32)
        self.receive_obs_reward(obs_rew, src=1)
        print(f"[EnvRunnerServerForNCCL] Received obs/reward: {obs_rew.tolist()}")
        return obs_rew.tolist()

    @override(EnvRunner)
    def assert_healthy(self):
        pass

    @override(EnvRunner)
    def get_metrics(self):
        return {}

    @override(EnvRunner)
    def stop(self):
        pass

    @override(Checkpointable)
    def get_ctor_args_and_kwargs(self):
        return (
            (),  # *args
            {"config": self.config},  # **kwargs
        )

    @override(Checkpointable)
    def get_checkpointable_components(self):
        return [
            (COMPONENT_RL_MODULE, self.module),
        ]

    @override(Checkpointable)
    def get_state(
        self,
        components: Optional[Union[str, Collection[str]]] = None,
        *,
        not_components: Optional[Union[str, Collection[str]]] = None,
        **kwargs,
    ) -> StateDict:
        return {
            COMPONENT_RL_MODULE: self.module.get_state(),
            WEIGHTS_SEQ_NO: self._weights_seq_no,
        }

    @override(Checkpointable)
    def set_state(self, state: StateDict) -> None:
        # Update the RLModule state.
        if COMPONENT_RL_MODULE in state:
            # A missing value for WEIGHTS_SEQ_NO or a value of 0 means: Force the
            # update.
            weights_seq_no = state.get(WEIGHTS_SEQ_NO, 0)

            # Only update the weigths, if this is the first synchronization or
            # if the weights of this `EnvRunner` lacks behind the actual ones.
            if weights_seq_no == 0 or self._weights_seq_no < weights_seq_no:
                rl_module_state = state[COMPONENT_RL_MODULE]
                if (
                    isinstance(rl_module_state, dict)
                    and DEFAULT_MODULE_ID in rl_module_state
                ):
                    rl_module_state = rl_module_state[DEFAULT_MODULE_ID]
                self.module.set_state(rl_module_state)

            # Update our weights_seq_no, if the new one is > 0.
            if weights_seq_no > 0:
                self._weights_seq_no = weights_seq_no

        if self._blocked_on_state is True:
            self._send_set_state_message()
            self._blocked_on_state = False

    def _client_message_listener(self):
        """Entry point for the listener thread."""

        # Set up the server socket and bind to the specified host and port.
        self._recycle_sockets()

        # Enter an endless message receival- and processing loop.
        while True:
            # As long as we are blocked on a new state, sleep a bit and continue.
            # Do NOT process any incoming messages (until we send out the new state
            # back to the client).
            if self._blocked_on_state is True:
                time.sleep(0.01)
                continue

            try:
                # Blocking call to get next message.
                msg_type, msg_body = get_rllink_message(self.client_socket)

                # Process the message received based on its type.
                # Initial handshake.
                if msg_type == RLlink.PING:
                    self._send_pong_message()

                # Episode data from the client.
                elif msg_type in [
                    RLlink.EPISODES,
                    RLlink.EPISODES_AND_GET_STATE,
                ]:
                    self._process_episodes_message(msg_type, msg_body)

                # Client requests the state (model weights).
                elif msg_type == RLlink.GET_STATE:
                    self._send_set_state_message()

                # Clients requests config information.
                elif msg_type == RLlink.GET_CONFIG:
                    self._send_set_config_message()

            except ConnectionError as e:
                print(f"Messaging/connection error {e}! Recycling sockets ...")
                self._recycle_sockets(5.0)
                continue

    def _recycle_sockets(self, sleep: float = 0.0):
        # Close all old sockets, if they exist.
        self._close_sockets_if_necessary()

        time.sleep(sleep)

        # Start listening on the configured port.
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow reuse of the address.
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        # Listen for a single connection.
        self.server_socket.listen(1)
        print(f"Waiting for client to connect to port {self.port}...")

        self.client_socket, self.address = self.server_socket.accept()
        print(f"Connected to client at {self.address}")

    def _close_sockets_if_necessary(self):
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()

    def _send_pong_message(self):
        send_rllink_message(self.client_socket, {"type": RLlink.PONG.name})

    def _process_episodes_message(self, msg_type, msg_body):
        # On-policy training -> we have to block until we get a new `set_state` call
        # (b/c the learning step is done and we can send new weights back to all
        # clients).
        if msg_type == RLlink.EPISODES_AND_GET_STATE:
            self._blocked_on_state = True

        episodes = []
        for episode_state in msg_body["episodes"]:
            episode = SingleAgentEpisode.from_state(episode_state)
            episodes.append(episode.to_numpy())

        # Push episodes into the to-be-returned list (for `sample()` requests).
        with self._sample_lock:
            if isinstance(self._episode_chunks_to_return, list):
                self._episode_chunks_to_return.extend(episodes)
            else:
                self._episode_chunks_to_return = episodes

    def _send_set_state_message(self):
        send_rllink_message(
            self.client_socket,
            {
                "type": RLlink.SET_STATE.name,
                "state": self.get_state(inference_only=True),
            },
        )

    def _send_set_config_message(self):
        send_rllink_message(
            self.client_socket,
            {
                "type": RLlink.SET_CONFIG.name,
                # TODO (sven): We need AlgorithmConfig to be a `Checkpointable` with a
                #  msgpack'able state.
                "config": pickle.dumps(self.config),
            },
        )

    def _log_episode_metrics(self, length, ret, sec):
        # Log general episode metrics.
        # To mimic the old API stack behavior, we'll use `window` here for
        # these particular stats (instead of the default EMA).
        win = self.config.metrics_num_episodes_for_smoothing
        self.metrics.log_value(EPISODE_LEN_MEAN, length, window=win)
        self.metrics.log_value(EPISODE_RETURN_MEAN, ret, window=win)
        self.metrics.log_value(EPISODE_DURATION_SEC_MEAN, sec, window=win)
        # Per-agent returns.
        self.metrics.log_value(
            ("agent_episode_returns_mean", DEFAULT_AGENT_ID), ret, window=win
        )
        # Per-RLModule returns.
        self.metrics.log_value(
            ("module_episode_returns_mean", DEFAULT_MODULE_ID), ret, window=win
        )

        # For some metrics, log min/max as well.
        self.metrics.log_value(EPISODE_LEN_MIN, length, reduce="min", window=win)
        self.metrics.log_value(EPISODE_RETURN_MIN, ret, reduce="min", window=win)
        self.metrics.log_value(EPISODE_LEN_MAX, length, reduce="max", window=win)
        self.metrics.log_value(EPISODE_RETURN_MAX, ret, reduce="max", window=win)

    @override(EnvRunner)
    def get_spaces(self):
        """
        Returns the observation and action spaces for the environment.
        """
        return self.config.get("observation_space"), self.config.get("action_space")