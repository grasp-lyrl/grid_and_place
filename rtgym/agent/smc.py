import torch
from typing import Union
from rtgym.agent.utils import hash_seed
from rtgym.dataclasses import AgentState, Trajectory


class SpatiallyModulatedCells:
    def __init__(self, arena, n_cells, sensory_key, sigma=8, magnitude=None, normalize=False, seed=None, device="cpu"):
        self.arena = arena
        self.n_cells = n_cells
        self.sigma = sigma
        self.magnitude = magnitude
        self.normalize = normalize
        self.sensory_key = sensory_key
        self.device = device

        self.seed = hash_seed(seed, sensory_key) if seed is not None else None
        self.rng = torch.Generator(device="cpu")  # CUDA generator has bugs, use CPU
        if self.seed is not None:
            self.rng.manual_seed(self.seed)  # Seed the rng

        assert self.n_cells > 0, "n_cells <= 0"
        assert isinstance(self.sigma, (int, float)) and self.sigma > 0, "sigma must be positive"

        self.sigma = self.sigma / self.arena.spatial_resolution

    def _init_response_map(self):
        print(f"  [{self.sensory_key}] Computing response map "
              f"({self.n_cells} cells, arena {list(self.arena.dimensions)})...")

        self.response_map = torch.zeros(
            (self.n_cells, *self.arena.dimensions), device=self.device
        )
        dims = self.arena.dimensions  # (H, W) or (D, H, W)

        # Generate noise on the arena grid
        cells = (
            torch.empty(self.n_cells, *dims)
            .normal_(0, 1, generator=self.rng)
            .to(self.device)
        )
        self.raw_field = cells.clone()  # Keep a copy of raw field

        # Boundary-aware diffusion smoothing
        cells = self.diffusion_smooth(cells)

        if self.magnitude is not None:
            cells = self._scale_to_magnitude(cells, self.magnitude)

        free_mask = (self.arena.map_ == 0).float().to(self.device)
        self.response_map = cells * free_mask
        print(f"  [{self.sensory_key}] Done.")

    @staticmethod
    def _scale_to_magnitude(cells, magnitude):
        # Compute the mean of the cells over the spatial dimensions
        cell_means = cells.mean(dim=tuple(range(1, cells.ndim)), keepdim=True)
        scaling_factors = magnitude / cell_means.clamp(
            min=1e-8
        )  # Avoid division by zero
        return cells * scaling_factors

    def diffusion_smooth(self, cells, free_mask=None):
        ndim = self.arena.ndim
        device = cells.device
        sigma = self.sigma

        if free_mask is None:
            free_mask = (self.arena.map_ == 0).float().to(device)

        # Diffusion iterations: effective σ ≈ sqrt(2n/3) for a box kernel,
        # so n ≈ 3σ²/2 to match the target sigma.
        n_iters = max(1, round(1.5 * sigma * sigma))

        # Small averaging kernel (applied as single-channel conv)
        if ndim == 2:
            kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
            pad_args = (1, 1, 1, 1)
            conv_fn = F.conv2d
        elif ndim == 3:
            kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
            pad_args = (1, 1, 1, 1, 1, 1)
            conv_fn = F.conv3d
        else:
            raise ValueError(f"Unsupported ndim={ndim}. Only 2D and 3D are supported.")

        # Precompute free-neighbour count (constant across iterations)
        mask_u = free_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, *dims)
        den = conv_fn(F.pad(mask_u, pad_args, mode='replicate'), kernel)[0, 0]
        den = den.clamp(min=1e-8)  # (*dims)

        # Zero out wall voxels in the initial field
        cells = cells * free_mask

        for _ in range(n_iters):
            x = cells.unsqueeze(1)  # (n, 1, *dims)
            x = F.pad(x, pad_args, mode='replicate')
            smoothed = conv_fn(x, kernel)[:, 0]  # (n, *dims)
            cells = (smoothed / den) * free_mask

        if self.normalize:
            spatial_dims = tuple(range(1, cells.ndim))
            cell_min = cells.amin(dim=spatial_dims, keepdim=True)
            cell_max = cells.amax(dim=spatial_dims, keepdim=True)
            cells = (cells - cell_min) / (cell_max - cell_min).clamp(min=1e-8)

        return cells

    def get_response(self, agent_data: Union[AgentState, Trajectory]):
        if isinstance(agent_data, Trajectory):
            idx = tuple(agent_data.int_coord[..., i] for i in range(self.arena.ndim))
            return self.response_map[(slice(None), *idx)].permute(1, 2, 0)
        elif isinstance(agent_data, AgentState):
            idx = tuple(agent_data.int_coord[:, i] for i in range(self.arena.ndim))
            return self.response_map[(slice(None), *idx)].permute(1, 0)
        else:
            raise ValueError(
                f"Invalid agent_data type: {type(agent_data)}, "
                "must be rtgym.data.Trajectory or rtgym.data.AgentState"
            )
