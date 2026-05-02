import torch
from typing import Optional


class AgentState:
    def __init__(
        self,
        coord=None,
        spd=None,
        spd_target=None,
        mv_dir=None,
        mv_dir_target=None,
        head_dir=None,
        device="cpu",
    ):
        self.device = torch.device(device)
        self._coord = coord
        self.spd = spd
        self.spd_target = spd_target
        self.mv_dir = mv_dir
        self.mv_dir_target = mv_dir_target
        self.head_dir = head_dir

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, value):
        self._coord = value.to(self.device) if value is not None else None

    @property
    def int_coord(self):
        return self._coord.long() if self._coord is not None else None

    def clone(self):
        new_state = AgentState(device=self.device)
        props = ["coord", "spd", "spd_target", "mv_dir", "mv_dir_target", "head_dir"]
        for prop in props:
            value = getattr(self, prop)
            if value is not None:
                setattr(new_state, prop, value.clone())
            else:
                setattr(new_state, prop, None)
        return new_state

    def reset(self):
        self._coord = None
        self.spd = None
        self.spd_target = None
        self.mv_dir = None
        self.mv_dir_target = None
        self.head_dir = None

    def to(self, device):
        self.device = torch.device(device)
        if self._coord is not None:
            self._coord = self._coord.to(device)
        if self.spd is not None:
            self.spd = self.spd.to(device)
        if self.spd_target is not None:
            self.spd_target = self.spd_target.to(device)
        if self.mv_dir is not None:
            self.mv_dir = self.mv_dir.to(device)
        if self.mv_dir_target is not None:
            self.mv_dir_target = self.mv_dir_target.to(device)
        if self.head_dir is not None:
            self.head_dir = self.head_dir.to(device)
        return self

    @property
    def disp(self):
        if self.spd is not None and self.mv_dir is not None:
            return self.spd * self.mv_dir
        return None

    def to_numpy(self):
        return (
            self.coord.cpu().numpy() if self.coord is not None else None,
            self.head_dir.cpu().numpy() if self.head_dir is not None else None,
            self.disp.cpu().numpy() if self.disp is not None else None,
        )



class Trajectory:
    def __init__(
        self,
        coord: Optional[torch.Tensor] = None,
        head_dir: Optional[torch.Tensor] = None,
        spd: Optional[torch.Tensor] = None,
        mv_dir: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self._coord = coord
        self.head_dir = head_dir
        self.spd = spd
        self.mv_dir = mv_dir

    def copy(self):
        return Trajectory(
            coord=self.coord, head_dir=self.head_dir, spd=self.spd,
            mv_dir=self.mv_dir, device=self.device
        )

    def __getitem__(self, index):
        def _process_index(idx, dim_size):
            """Convert index to torch.Tensor for advanced indexing"""
            if isinstance(idx, int):
                return torch.tensor([idx], device=self.device)
            elif isinstance(idx, slice):
                return torch.arange(dim_size, device=self.device)[idx]
            elif isinstance(idx, (list, tuple)):
                return torch.tensor(idx, device=self.device)
            elif isinstance(idx, torch.Tensor):
                return idx.to(self.device)
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")

        # Handle single index (batch dimension indexing)
        if not isinstance(index, tuple):
            batch_idx = _process_index(index, self._coord.shape[0])
            return Trajectory(
                coord=self._coord[batch_idx, :, :],
                head_dir=self.head_dir[batch_idx, :, :],
                spd=self.spd[batch_idx, :, :],
                mv_dir=self.mv_dir[batch_idx, :, :],
                device=self.device,
            )

        # Handle tuple indexing (batch_idx, time_idx)
        batch_idx, time_idx = index
        batch_size, time_size = self._coord.shape[:2]

        # Process indices
        if batch_idx is Ellipsis or batch_idx == slice(None):
            batch_tensor = torch.arange(batch_size, device=self.device)
        else:
            batch_tensor = _process_index(batch_idx, batch_size)

        if time_idx is Ellipsis or time_idx == slice(None):
            time_tensor = torch.arange(time_size, device=self.device)
        else:
            time_tensor = _process_index(time_idx, time_size)

        # Use advanced indexing
        batch_mesh, time_mesh = torch.meshgrid(batch_tensor, time_tensor, indexing="ij")

        # Index all tensors
        coord_result = self._coord[batch_mesh, time_mesh, :]
        head_dir_result = self.head_dir[batch_mesh, time_mesh, :]
        spd_result = self.spd[batch_mesh, time_mesh, :]
        mv_dir_result = self.mv_dir[batch_mesh, time_mesh, :]

        # Return AgentState for single timestep, Trajectory otherwise
        if coord_result.shape[1] == 1 and coord_result.shape[0] >= 1:
            return AgentState(
                coord=coord_result.squeeze(1),
                head_dir=head_dir_result.squeeze(1),
                spd=spd_result.squeeze(1),
                mv_dir=mv_dir_result.squeeze(1),
                device=self.device,
            )
        else:
            return Trajectory(
                coord=coord_result,
                head_dir=head_dir_result,
                spd=spd_result,
                mv_dir=mv_dir_result,
                device=self.device,
            )

    def __len__(self):
        if self._coord is None:
            raise ValueError("Coordinate data is not set")
        return self._coord.shape[1]

    def slice(self, start, end):
        return self.t_range((start, end))

    @property
    def size(self):
        return self._coord.shape[0], self._coord.shape[1]

    @property
    def int_coord(self):
        return self._coord.long() if self._coord is not None else None

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, value):
        self._coord = value

    @property
    def disp(self):
        return self.spd * self.mv_dir
