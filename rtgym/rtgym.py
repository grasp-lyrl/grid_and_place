import torch
from rtgym.arena import Arena
from rtgym.agent import Agent


class RatatouGym:
    def __init__(self, temporal_resolution, spatial_resolution, device="cpu"):
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.device = torch.device(device)
        self.arena = Arena(self, device=device)
        self.agent = Agent(self, device=device)

    def _on_arena_change(self):
        self.agent._on_arena_change()

    def init_arena_map(self, **kwargs):
        self.arena.init_arena_map(**kwargs)

    @property
    def t_res(self):
        return self.temporal_resolution
