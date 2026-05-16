import torch
import numpy as np


class RatemapAggregatorEMA:
    def __init__(self, arena_map, decay, device):
        assert isinstance(arena_map, torch.Tensor), (
            "RatemapAggregator only takes tensor; arena_map must be a tensor"
        )
        self.arena_map = arena_map
        self.dims = arena_map.shape  # (n_x, n_y) or (n_x, n_y, n_z)
        self.n_cells = None
        self.device = torch.device(device)
        self.decay = decay

    def init_counts(self):
        self.partial_sums = torch.zeros(
            (self.n_cells, *self.dims), dtype=torch.float32, device=self.device
        )
        # shape: (n_x, n_y) or (n_x, n_y, n_z)
        self.visit_counts = torch.zeros(
            self.dims, dtype=torch.float32, device=self.device
        )

    def get_ratemap(self):
        # Avoid division by zero by clamping
        denom = self.visit_counts.clamp(min=1.0)  # shape: (n_x, n_y)

        # Broadcasting: partial_sums shape (n_cells, *dims) / (*dims)
        ratemap = self.partial_sums / denom.unsqueeze(0)

        # Set unvisited areas to NaN
        mask_unvisited = self.visit_counts == 0
        ratemap[:, mask_unvisited] = float("nan")

        return ratemap

    def update(self, states, coords):
        assert isinstance(states, torch.Tensor), (
            "RatemapAggregator only takes tensor; states must be a tensor"
        )
        if self.n_cells is None:
            self.n_cells = states.shape[-1]
            self.init_counts()

        states = states.to(self.device).float()
        coords = torch.round(coords.to(self.device)).long()

        ndim = coords.shape[-1]  # 2 or 3
        coords = coords.reshape(-1, ndim)  # flatten the batch x time
        states = states.reshape(-1, self.n_cells)  # flatten the batch x time

        flat_sums = self.partial_sums.view(self.n_cells, -1)
        flat_counts = self.visit_counts.view(-1)

        # Compute flat index for nd coordinates
        flat_coords = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        stride = 1
        for i in reversed(range(ndim)):
            flat_coords = flat_coords + coords[:, i] * stride
            stride *= self.dims[i]

        # Get unique bins visited in this batch
        visited_bins = flat_coords.unique()

        # Decay only the visited bins
        flat_sums[:, visited_bins] *= self.decay
        flat_counts[visited_bins] *= self.decay

        # Accumulate new data
        flat_sums.index_add_(1, flat_coords, states.T)
        flat_counts.index_add_(
            0, flat_coords, torch.ones_like(flat_coords, dtype=torch.float32)
        )

    def reset(self):
        self.partial_sums.zero_()
        self.visit_counts.zero_()