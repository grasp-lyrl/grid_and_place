import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from omegaconf import DictConfig, OmegaConf

from model import RNN
from rtgym import RatatouGym
from utils.misc import setup_experiment
from utils.ratemap_aggregator import RatemapAggregatorEMA


class RectRoomEnv:
    def __init__(self, env_config, trajectory_len, masking_ratio, device):
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

        self.sens_dim = self.agent.num_neurons(keys=self.sens_keys)
        self._neuron_profiles = env_config.neuron_profiles

        self.mean_fr = torch.cat([
            self.agent.neurons.neuron_groups[k].response_map.mean(dim=(1, 2))
            for k in self.sens_keys
        ], dim=0)
        self.masking_ratio = masking_ratio

        self._gen_len = int(env_config.n_traj * trajectory_len)
        self._trajectory_len = trajectory_len
        self._trajectory = None
        self._ptr = 0

    def masking_fn(self, x):
        B, T, _ = x.shape
        mask = torch.rand(B, T, 1, device=x.device) < self.masking_ratio
        return torch.where(mask, self.mean_fr, x)

    def sample(self, batch_size):
        # Generate/re-generate trajectory if missing or exhausted
        if (
            self._trajectory is None
            or self._ptr + self._trajectory_len >= self._trajectory.coord.shape[1]
        ):
            self._trajectory = self.agent.random_traverse(self._gen_len, batch_size)
            self._ptr = 0

        traj_slice = self._trajectory[:, self._ptr:self._ptr + self._trajectory_len]
        self._ptr += self._trajectory_len

        # Sample motion and sensory input
        m_seq = torch.cat([traj_slice.mv_dir, traj_slice.spd], dim=-1)[:, 1:]
        s_seq = self.agent.get_response(traj_slice, keys=self.sens_keys)[:, :-1]
        coord_seq = traj_slice.coord[:, :-1]

        # Construct input/label
        m_in = m_seq[:, :-1]
        s_in = self.masking_fn(s_seq[:, :-1])
        s_target = s_seq[:, 1:]

        return s_in, m_in, s_target, coord_seq[:, 1:]


class Logger:
    def __init__(self, arena_map, save_dir, cfg_logging, device, model):
        self.model = model
        self.save_dir = save_dir
        self.save_every = cfg_logging.save_every
        self.log_every = cfg_logging.log_every

        self._arena_map = arena_map
        self._device = device
        self._ds_ratio = cfg_logging.downsample_ratio
        self.rma = RatemapAggregatorEMA(
            arena_map=arena_map, device=device, decay=cfg_logging.rma_decay
        )

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = None

    def step(self, hidden_seq, coords, loss, step):
        self.rma.update(states=hidden_seq, coords=coords)

        if step % self.log_every == 0:
            print(f"Step {step}: Loss = {loss:.4f}")

        if  step % self.save_every == 0:
            if self._pending is not None:
                self._pending.result()

            ratemaps = self.rma.get_ratemap()
            ratemaps_ds = F.avg_pool2d(
                ratemaps.unsqueeze(0), 
                kernel_size=self._ds_ratio
            ).squeeze(0).cpu()

            for region, sl in [("clamped", self.model.clamped_slice),
                               ("free", self.model.free_slice)]:
                self._executor.submit(
                    np.savez_compressed,
                    self.save_dir / f"ratemap_{region}_step_{step}.npz",
                    ratemap=ratemaps_ds[sl].numpy(),
                )


def train(rnn, env, logger, scaler, optimizer, training_steps, batch_size):
    rnn.train()
    h, step = None, 0

    while step < training_steps:
        s_in, m_in, s_target, coord_seq = env.sample(batch_size)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            hidden_seq, pred_seq = rnn(s_in, m_in, h_init=h)
            loss = F.mse_loss(pred_seq, s_target)
            h = hidden_seq[:, -1, :].detach()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        rnn.rnn_cell.project_ei_homeostasis_()

        step += s_in.shape[1]
        logger.step(
            hidden_seq=hidden_seq.detach(),
            coords=coord_seq.detach(),
            loss=loss.item(),
            step=step,
        )

    return step


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
        trajectory_len=cfg.training.trajectory_len,
        masking_ratio=cfg.training.masking_ratio,
        device=device,
    )

    # Model
    rnn = RNN(
        d_clamped=cfg.model.d_clamped,
        d_free=cfg.model.d_free,
        input_dim=env.sens_dim,
        alpha=cfg.model.alpha,
        noise_level=cfg.model.noise_level,
        motion_dim=3 if cfg.model.use_motion else 0,
        homeostasis_eta=cfg.training.homeostasis_eta,
    )
    rnn.to(device)

    optimizer = torch.optim.AdamW(rnn.parameters(), lr=cfg.training.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    step = 0

    logger = Logger(
        arena_map=env.arena.map_,
        save_dir=save_dir,
        cfg_logging=cfg.logging,
        device=device,
        model=rnn,
    )

    train(
        rnn=rnn,
        env=env,
        logger=logger,
        scaler=scaler,
        optimizer=optimizer,
        training_steps=cfg.training.training_steps,
        batch_size=cfg.training.batch_size,
    )


if __name__ == "__main__":
    main()
