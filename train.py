import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from omegaconf import DictConfig, OmegaConf

from rtgym import RatatouGym
from utils.misc import setup_experiment


class RectRoomEnv:
    def __init__(self, env_config, encode_horizon, device):
        self.gym = RatatouGym(
            temporal_resolution=env_config.temporal_resolution,
            spatial_resolution=env_config.spatial_resolution,
            device=device,
        )
        self.agent = self.gym.agent
        self.arena = self.gym.arena

        print("\033[96mInitializing arena map...\033[0m")
        self.gym.init_arena_map(**env_config.arena_config)
        self.agent.init_control(env_config.control_config)
        self.agent.init_neurons(env_config.neuron_profiles)
        self.sens_keys = list(env_config.neuron_profiles.keys())
        print("\033[92mGym initialized.\033[0m")


def main():
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="default", help="Config name in configs/"
    )
    parser.add_argument(
        "--name", type=str, help="Run name"
    )
    parser.add_argument(
        "--save_dir", type=str, default="default", help="Root save directory"
    )
    args, overrides = parser.parse_known_args()
    cfg, save_dir = setup_experiment(args, overrides)

    # Set seed
    np.random.seed(cfg.globle_seed)
    torch.manual_seed(cfg.globle_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.globle_seed)
    # Set the seed for the neuron_profiles.
    for key in cfg.env.neuron_profiles:
        cfg.env.neuron_profiles[key].seed = cfg.globle_seed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment
    env = RectRoomEnv(
        env_config=cfg.env,
        encode_horizon=cfg.training.encode_horizon,
        device=device,
    )

if __name__ == "__main__":
    main()
