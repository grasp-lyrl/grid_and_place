import torch
from typing import Dict, Any, Union, List
from rtgym.dataclasses import AgentState, Trajectory
from rtgym.agent.smc import SpatiallyModulatedCells


class Neurons:
    def __init__(self, gym, device="cpu"):
        self.gym = gym
        self.device = device
        self.neuron_groups = {}
        self.ranges = None  # Keep track of the indices of the simulated neuronal groups

    def init_from_profile(self, neuron_profiles):
        self.neuron_groups = {}
        self._update_neurons(neuron_profiles if neuron_profiles is not None else {})
        self._update_ranges()

    def _update_neurons(self, profile_list: Dict[str, Any]):
        for key, value in profile_list.items():
            self.neuron_groups[key] = SpatiallyModulatedCells(
                sensory_key=key, device=self.device, arena=self.gym.arena, **value
            )

    def _update_ranges(self):
        cell_counts = torch.tensor(
            [_sens.n_cells for _sens in self.neuron_groups.values()]
        )
        _ranges = torch.cumsum(cell_counts, 0)
        _ranges = torch.cat([torch.zeros(1, dtype=_ranges.dtype), _ranges]).tolist()
        self.ranges = {
            key: (_ranges[i], _ranges[i + 1])
            for i, key in enumerate(self.neuron_groups.keys())
        }

    def filter_neurons(self, keys=None, str_filter=None, type_filter=None):
        if keys is not None:
            if isinstance(keys, str):
                return_keys = [keys]
            elif hasattr(keys, "__iter__"):
                return_keys = list(keys)
            else:
                raise ValueError(f"Unknown keys type: {type(keys)}")
        elif str_filter is not None:
            return_keys = [
                key for key in self.neuron_groups.keys() if str_filter in key
            ]
        elif type_filter is not None:
            return_keys = [
                key
                for key, neurons in self.neuron_groups.items()
                if type_filter == neurons.neuron_type
            ]
        else:
            return_keys = list(self.neuron_groups.keys())
        return sorted(return_keys)

    def num_neurons(self, keys=None, str_filter=None, type_filter=None):
        keys = self.filter_neurons(keys, str_filter, type_filter)
        return sum([self.neuron_groups[key].n_cells for key in keys])

    def get_response(
        self,
        agent_data: Union[AgentState, Trajectory],
        return_format: str = "tensor",
        keys: List[str] = None,
        str_filter: str = None,
        type_filter: str = None,
    ):
        # Set filter_keys if not provided
        keys = self.filter_neurons(keys, str_filter, type_filter)
        if return_format == "dict":
            return {
                key: self.neuron_groups[key].get_response(agent_data) for key in keys
            }
        elif return_format == "tensor":
            res_list = []
            for key in keys:
                res = self.neuron_groups[key].get_response(agent_data)
                res_list.append(res)
            return torch.cat(res_list, dim=-1)
        else:
            raise ValueError(f"Unknown return format: {return_format}")
