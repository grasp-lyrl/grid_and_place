import os
import torch
import matplotlib.pyplot as plt
from .arena_shapes import *

class Arena:
    def __init__(self, gym, device="cpu"):
        self.gym = gym
        self.spatial_resolution = gym.spatial_resolution
        self.device = torch.device(device)
        self._arena_map = None

        # Initialize compiled validation methods
        self._validate_index_compiled = None
        self._validate_index_cpu_compiled = None
        self._compile_validation_methods()

    def _compile_validation_methods(self):
        disable_compile = os.environ.get("RTGYM_DISABLE_COMPILE", "0") == "1"
        if hasattr(torch, "compile") and not disable_compile:
            try:
                compile_opts = {
                    "fullgraph": True,
                    "options": {
                        "triton.cudagraphs": False,
                        "epilogue_fusion": False,
                        "max_autotune": False,
                    },
                }
                self._validate_index_compiled = torch.compile(
                    self._validate_index_kernel, **compile_opts
                )
                self._validate_index_cpu_compiled = torch.compile(
                    self._validate_index_cpu_kernel, **compile_opts
                )
            except Exception as e:
                print(f"Failed to compile arena validation methods: {e}")

    def _warmup_validation_methods(self):
        if hasattr(self, "_free_flat") and self._free_flat is not None:
            batch_size = 2
            ndim = self.ndim
            test_pos = torch.rand(batch_size, ndim, device=self.device)
            for i in range(ndim):
                test_pos[:, i] *= self.dimensions[i] - 1
            test_pos = test_pos.long()

            if self._validate_index_compiled:
                _ = self._validate_index_compiled(test_pos)
            if self._validate_index_cpu_compiled:
                _ = self._validate_index_cpu_compiled(test_pos.cpu())

    @property
    def ndim(self):
        return len(self.dimensions)

    @property
    def map_(self):
        return self._arena_map

    @map_.setter
    def map_(self, arena_map):
        assert isinstance(arena_map, torch.Tensor), "Arena map must be a torch tensor."
        self._arena_map = arena_map.to(self.device)

        self.dimensions = self._arena_map.shape
        self.free_space = torch.nonzero(self._arena_map == 0)

        # For fast validation (put to cpu because generate behavior is much faster on cpu)
        self._free_flat = (self._arena_map == 0).reshape(-1).contiguous().to("cpu")

        # Compute strides for nd flat indexing: e.g., 2D: [W, 1], 3D: [H*W, W, 1]
        strides = []
        s = 1
        for d in reversed(arena_map.shape):
            strides.append(s)
            s *= d
        self._strides = list(reversed(strides))

        # Warmup compiled validation methods with new arena
        self._warmup_validation_methods()

        # Notify gym of arena change
        if hasattr(self.gym, "_on_arena_change"):
            self.gym._on_arena_change()

    @property
    def invmap_(self):
        return 1 - self._arena_map

    def set_arena_map(self, arena_map):
        assert isinstance(arena_map, torch.Tensor), "arena map must be a torch tensor"
        assert arena_map.ndim == 2, (
            "set_arena_map currently only supports 2D maps. "
            f"Got {arena_map.ndim}D tensor with shape {arena_map.shape}."
        )

        # Check if its edges are all 1. If not, pad them with 1
        if torch.all(arena_map[0, :] == 0):
            arena_map = torch.nn.functional.pad(arena_map, (0, 0, 1, 0), value=1)
        if torch.all(arena_map[-1, :] == 0):
            arena_map = torch.nn.functional.pad(arena_map, (0, 0, 0, 1), value=1)
        if torch.all(arena_map[:, 0] == 0):
            arena_map = torch.nn.functional.pad(arena_map, (1, 0, 0, 0), value=1)
        if torch.all(arena_map[:, -1] == 0):
            arena_map = torch.nn.functional.pad(arena_map, (0, 1, 0, 0), value=1)

        self.map_ = arena_map

    def init_arena_map(self, shape, **kwargs):
        shape_generators = {
            "rectangle": generate_rectangle_arena,
            "hairpin": generate_hairpin_arena,
            "carpenter_rooms": generate_carpenter_rooms_arena,
            "box": generate_box_arena,
        }

        if shape not in shape_generators:
            raise ValueError(
                f"Unknown shape '{shape}'. Valid options are: {list(shape_generators.keys())}"
            )

        # Generate the arena map
        self.map_ = shape_generators[shape](self.spatial_resolution, **kwargs)

    def generate_random_pos(self, batch_size: int):
        indices = torch.randint(
            0, self.free_space.shape[0], (batch_size,), device=self.device
        )
        return self.free_space[indices]

    def _validate_index_kernel(self, pos: torch.Tensor) -> torch.Tensor:
        inb, flat, device = self._validate_index_common(pos)
        free = torch.take(self._free_flat.to(device), flat)
        return inb & free

    def _validate_index_cpu_kernel(self, pos: torch.Tensor) -> torch.Tensor:
        inb, flat, _ = self._validate_index_common(pos)
        free = torch.take(self._free_flat, flat)
        return inb & free

    def _validate_index_common(self, pos: torch.Tensor):
        ndim = self.ndim
        pos = pos.view(-1, ndim)

        # Check bounds for each dimension
        inb = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
        for i in range(ndim):
            inb = inb & (pos[:, i] >= 0) & (pos[:, i] < self.dimensions[i])

        # Clamp and compute flat index
        flat = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
        for i in range(ndim):
            pos[:, i].clamp_(0, self.dimensions[i] - 1)
            flat = flat + pos[:, i] * self._strides[i]

        return inb, flat, pos.device

    def validate_index(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Validate whether given positions are within arena bounds and free space.
        Returns a boolean mask for validity (not out of bounds, not in wall).
        """
        if self._validate_index_compiled:
            return self._validate_index_compiled(pos)
        return self._validate_index_kernel(pos)

    # def validate_index(self, pos: torch.Tensor) -> torch.Tensor:
    #     if self._validate_index_compiled:
    #         return self._validate_index_compiled(pos)
    #     return self._validate_index_kernel(pos)

    # def validate_index_cpu(self, pos: torch.Tensor) -> torch.Tensor:
    #     if self._validate_index_cpu_compiled:
    #         return self._validate_index_cpu_compiled(pos)
    #     return self._validate_index_cpu_kernel(pos)
