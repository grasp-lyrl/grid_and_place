import torch
from typing import Union
from rtgym.agent.control import TrajectoryGenerator
from rtgym.agent.neurons import Neurons
from rtgym.dataclasses import AgentState, Trajectory


class Agent:
    def __init__(self, gym, device="cpu"):
        self.gym = gym
        self.device = device
        self.control_profile = None
        self.neuron_profiles = None
        self.neurons = Neurons(self.gym, device=device)
        self.control = None

    @property
    def neuron_groups(self):
        return self.neurons.neuron_groups
    
    def num_neurons(self, keys=None, str_filter=None, type_filter=None):
        return self.neurons.num_neurons(keys, str_filter, type_filter)

    @property
    def arena(self):
        return self.gym.arena

    @property
    def state(self):
        if self.control is None:
            raise ValueError("Control is not initialized, call init_control() first.")
        return self.control.cur_state

    def _on_arena_change(self):
        if self.arena is not None:
            # Reinitialize control from profile
            if self.control is not None:
                self.control._on_arena_change()
                self.control = TrajectoryGenerator(self.gym)
                self.control.init_from_profile(self.control_profile)

            # Reinitialize neurons from profile
            if self.neurons is not None:
                self._init_neurons_from_profile()

    def init_control(self, control_profile: dict):
        self.control_profile = control_profile
        self.control = TrajectoryGenerator(self.gym)
        self.control.init_from_profile(self.control_profile)

    def init_neurons(self, neuron_profiles: dict):
        self.neuron_profiles = neuron_profiles
        self._init_neurons_from_profile()

    def add_neuron_group(self, neuron_profile: dict):
        assert neuron_profile is not None, "neuron_profile is None"
        self.neuron_profiles.update(neuron_profile)
        self.neurons.add_neuron_group(neuron_profile)

    def _init_neurons_from_profile(self):
        if self.neuron_profiles is not None and self.arena is not None:
            self.neurons.init_from_profile(self.neuron_profiles)

    def random_traverse(
        self, duration: float, batch_size: int, init_state=None, pause_prob=0, **kwargs
    ):
        # Continue from current state if no explicit init_state
        if init_state is None and self.state is not None and self.state.coord is not None:
            init_state = self.state

        traj, state = self.control.generate_trajectory(duration, batch_size, init_state)

        # Update agent state for continuation
        self.control.cur_state = state

        if pause_prob > 0:
            pause_mask = torch.rand(batch_size, device=self.device) < pause_prob
            traj.disp[pause_mask] = torch.zeros_like(traj.disp[pause_mask])
            traj.coord[pause_mask] = traj.coord[pause_mask, 0].unsqueeze(1)
            traj.head_dir[pause_mask] = torch.zeros_like(traj.head_dir[pause_mask])
        return traj

    def step(
        self, mv_dir: torch.Tensor, spd: torch.Tensor, head_dir: torch.Tensor
    ) -> None:
        return self.control.step(self.control.cur_state, mv_dir, spd, head_dir)

    def step_state(
        self, state: AgentState, mv_dir: torch.Tensor, 
        spd: torch.Tensor, head_dir: torch.Tensor
    ) -> None:
        return self.control.step(state, mv_dir, spd, head_dir)

    def get_response(
        self, agent_data: Union[AgentState, Trajectory], return_format: str = "tensor",
        keys: list = None, str_filter: str = None, type_filter: str = None,
    ):
        return self.neurons.get_response(
            agent_data, return_format, keys, str_filter, type_filter
        )

    def spawn(self, init_state=None) -> None:
        self.control.reset()
        if init_state is not None:
            if isinstance(init_state, AgentState):
                self.control.cur_state = init_state.clone()
            else:
                raise ValueError(
                    f"Invalid init_state type: {type(init_state)}, "
                    "must be rtgym.data.AgentState"
                )
        else:
            raise ValueError(
                "init_state must be provided, or use random_spawn() "
                "to spawn at a random position."
            )
        return self.control.cur_state

    def random_spawn(self, batch_size: int):
        init_state = AgentState(device=self.device)
        init_state.coord = (
            self.arena.generate_random_pos(batch_size).float().to(self.device)
        )
        self.spawn(init_state)
