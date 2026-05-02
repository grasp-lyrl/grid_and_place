import os
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from rtgym.dataclasses import AgentState, Trajectory
from rtgym.agent.utils import TrajectoryGeneratorConfig, TrajectoryBuilder


class BaseControl:
    compile_options = {
        "fullgraph": True,
        "options": {
            "triton.cudagraphs": False,
            "epilogue_fusion": False,
            "max_autotune": False,
        },
    }

    def __init__(self, gym, bhv_device):
        self.gym = gym
        self.bhv_device = torch.device(bhv_device)
        self._coord_update_compiled = None
        self.cur_state = AgentState(device=self.bhv_device)

        # Compile coordinate update kernel if available and not disabled
        disable_compile = os.environ.get("RTGYM_DISABLE_COMPILE", "0") == "1"
        if hasattr(torch, "compile") and not disable_compile:
            self._coord_update_compiled = torch.compile(
                self._coord_update_kernel, **self.compile_options
            )

        # Warmup coordinate update kernel
        if self._coord_update_compiled:
            batch_size = 2
            ndim = 2  # Safe default; actual ndim comes from arena at runtime
            coord = torch.randn(batch_size, ndim, device=self.bhv_device)
            step = torch.randn(batch_size, ndim, device=self.bhv_device)
            _ = self._coord_update_compiled(coord, step)

    def _coord_update_kernel(self, coord, disp):
        """Optimized coordinate update kernel."""
        return coord + disp

    def _update_coord(self, raw_agent_state):
        """
        Optimized coordinate update with boundary collision detection.
        Uses compiled kernel when available for better CPU performance.
        """
        # Use compiled kernel if available
        _coord_kernel = self._coord_update_compiled or self._coord_update_kernel
        updated_coord = _coord_kernel(raw_agent_state.coord, raw_agent_state.disp)

        # Fast boundary check and rejection
        validate_index = (
            self.gym.arena.validate_index_cpu
            if self.bhv_device == "cpu"
            else self.gym.arena.validate_index
        )
        invalid_mask = ~validate_index(updated_coord.long())
        if invalid_mask.any():
            updated_coord[invalid_mask] = raw_agent_state.coord[invalid_mask]
        raw_agent_state.coord = updated_coord


class TrajectoryGenerator(BaseControl):
    def __init__(self, gym, bhv_device=None):
        if bhv_device is None:
            bhv_device = gym.device
        super().__init__(gym, bhv_device=bhv_device)

        # Initialize compiled kernel attributes
        self.initialized = False
        self._phi_n_compiled = None
        self._ema_step_compiled = None

        # Initialize behavior configuration
        self.cfg = TrajectoryGeneratorConfig()

        # Compile EMA step function if not disabled
        disable_compile = os.environ.get("RTGYM_DISABLE_COMPILE", "0") == "1"
        if hasattr(torch, "compile") and not disable_compile:
            self._ema_step_compiled = torch.compile(
                self._ema_update_step, **self.compile_options
            )

    def init_from_profile(self, raw_profile):
        try:
            self.cfg = TrajectoryGeneratorConfig.from_profile(raw_profile, self.gym)
            self._recompute_maps()
            self.initialized = True
        except ValueError as e:
            print(f"Autonomous behavior not initialized: {e}")
            self.initialized = False

    def generate_trajectory(
        self, duration: float, batch_size: int, init_state=None
    ) -> tuple[Trajectory, AgentState]:
        if not self.initialized:
            raise ValueError(
                "Autonomous behavior not initialized. Check if the arena is set properly.",
                "Call init_from_profile() to initialize the behavior.",
            )

        ndim = self.gym.arena.ndim
        B, T = batch_size, self.gym.to_ts(duration)
        traj_builder = TrajectoryBuilder(
            max_length=T, batch_size=B, ndim=ndim, device=self.bhv_device
        )

        # Precompute all random targets including head directions
        self._precompute_random_targets(B, T)

        # Initialize trajectory with starting agent state
        self.cur_state = self._init_agent_state(batch_size, init_state)
        traj_builder.set_state(0, self.cur_state)

        # Simulate all trajectory time steps
        self._simulate_trajectory_steps(traj_builder, T)

        return traj_builder.finalize(), self.cur_state

    def _precompute_random_targets(self, batch_size, time_steps):
        B, T = batch_size, time_steps
        ndim = self.gym.arena.ndim

        # Speed switching and targets (unchanged — speed is scalar)
        self.switch_spd_mask = (
            torch.rand((B, T), device=self.bhv_device) < self.cfg.switch_spd_prob
        )
        self.target_spds = (
            self._generate_random_speeds(
                size=(B, T, 1), spd_mean=self.cfg.spd_mean, spd_sd=self.cfg.spd_sd
            )
            / 1e3
            * self.gym.t_res
        )

        # Direction switching and targets — ndim-agnostic
        self.switch_mv_dir_mask = (
            torch.rand((B, T), device=self.bhv_device) < self.cfg.switch_dir_prob
        )
        dirs = torch.randn((B, T, ndim), device=self.bhv_device)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        self.target_mv_dirs = dirs

        # Precompute head directions for entire trajectory
        self.head_dirs = self._generate_random_head_dirs(dirs, B, T)

    def _simulate_trajectory_steps(self, traj_builder, time_steps):
        """Simulate all trajectory time steps."""
        boundary_params = self._setup_boundary_avoidance()
        ema_kernel = self._ema_step_compiled or self._ema_update_step

        for ts in range(1, time_steps):
            self._update_agent_targets(ts)
            self._apply_ema_dynamics(ema_kernel)
            self._apply_boundary_avoidance(boundary_params)
            self._update_coord(self.cur_state)
            # Update head direction from precomputed values
            self.cur_state.head_dir = self.head_dirs[:, ts]
            traj_builder.set_state(ts, self.cur_state)

    def _setup_boundary_avoidance(self):
        """Setup boundary avoidance parameters if enabled."""
        if self.cfg.avoid_boundary_dist <= 0 or not hasattr(self, "distance_map"):
            return None
        return {"enabled": True}

    def _update_agent_targets(self, timestep):
        """Update agent speed and direction targets at given timestep."""
        spd_mask = self.switch_spd_mask[:, timestep]
        if spd_mask.any():
            self.cur_state.spd_target[spd_mask] = self.target_spds[spd_mask, timestep]

        dir_mask = self.switch_mv_dir_mask[:, timestep]
        if dir_mask.any():
            self.cur_state.mv_dir_target[dir_mask] = self.target_mv_dirs[
                dir_mask, timestep
            ]

    def _apply_ema_dynamics(self, ema_kernel):
        """Apply exponential moving average dynamics to speed and direction."""
        a_s, a_d = self.cfg.alpha_spd, self.cfg.alpha_dir
        self.cur_state.spd, self.cur_state.mv_dir = ema_kernel(
            self.cur_state.spd,
            self.cur_state.mv_dir,
            self.cur_state.spd_target,
            self.cur_state.mv_dir_target,
            a_s,
            a_d,
        )

    def _apply_boundary_avoidance(self, boundary_params):
        """Boundary avoidance using vector projection onto wall tangent plane."""
        if boundary_params is None or not boundary_params["enabled"]:
            return

        coord = self.cur_state.coord
        ndim = coord.shape[-1]

        # Integer coordinates for map lookup, clamped to bounds
        int_coord = tuple(
            coord[:, i].long().clamp(0, self._boundary_dims[i] - 1)
            for i in range(ndim)
        )

        avoid_coef = self.distance_map[int_coord]  # (B,)
        active = avoid_coef > 1e-2
        if not active.any():
            return

        u = self.cur_state.mv_dir[active]  # (n, ndim)

        # Wall normal at active positions (points away from wall)
        active_coord = tuple(ic[active] for ic in int_coord)
        normal = self.wall_normal_map[active_coord]  # (n, ndim)

        # Dot product: negative means heading toward wall
        dot = (u * normal).sum(dim=-1)  # (n,)
        pointing_toward = (dot < 0).float()

        # Project onto tangent plane: v' = v - (v·n)*n
        v_tangent = u - dot.unsqueeze(-1) * normal
        v_tangent_norm = v_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v_tangent = v_tangent / v_tangent_norm

        # Speed-scaled blending coefficient
        spd = self.cur_state.spd[active].squeeze(-1)  # (n,)
        mean_spd = self.cfg.spd_mean / 1e3 * self.gym.t_res
        spd_scale = (spd / (mean_spd + 1e-8)).clamp(0.5, 3.0)

        blend = (avoid_coef[active] * pointing_toward * spd_scale).unsqueeze(-1)  # (n, 1)

        # Blend between current direction and tangent projection
        new_dir = (1 - blend) * u + blend * v_tangent
        new_dir = new_dir / new_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        self.cur_state.mv_dir[active] = new_dir

    def _ema_update_step(
        self, cur_spd, cur_mv_dir, target_spd, target_mv_dir, a_s, a_d
    ):
        """Optimized EMA update step - compiled for performance."""
        new_spd = (1.0 - a_s) * cur_spd + a_s * target_spd
        new_mv_dir = (1.0 - a_d) * cur_mv_dir + a_d * target_mv_dir
        new_mv_dir = new_mv_dir / new_mv_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return new_spd, new_mv_dir

    def _init_agent_state(self, batch_size: int, init_state=None):
        """Initialize agent state for a batch."""
        agent_state = (
            AgentState(device=self.bhv_device)
            if init_state is None
            else init_state.clone()
        )

        # Fill any missing fields as needed
        if agent_state.coord is None:
            agent_state.coord = (
                self.gym.arena.generate_random_pos(batch_size)
                .float()
                .to(self.bhv_device)
            )
        if agent_state.spd is None or agent_state.spd_target is None:
            agent_state.spd = self.target_spds[:, 0]
            agent_state.spd_target = self.target_spds[:, 0]
        if agent_state.mv_dir is None or agent_state.mv_dir_target is None:
            agent_state.mv_dir = self.target_mv_dirs[:, 0]
            agent_state.mv_dir_target = self.target_mv_dirs[:, 0]
        if agent_state.head_dir is None:
            agent_state.head_dir = self.head_dirs[:, 0]

        return agent_state

    # ===================================== Helper Functions =====================================
    def _on_arena_change(self):
        """
        Handle arena change events.
        """
        self._recompute_maps()

    def _generate_random_speeds(
        self, size: tuple, spd_mean: float, spd_sd: float
    ) -> torch.Tensor:
        # Unpack size and parameters
        s, m = spd_sd, spd_mean

        # If spd_sd > 0, generate log-normal distributed speeds
        if spd_sd > 0:
            sigma = (
                torch.log(torch.tensor(1 + (s**2 / m**2), device=self.bhv_device))
                ** 0.5
            ).item()
            mu = torch.log(
                torch.tensor(m**2 / (m**2 + s**2) ** 0.5, device=self.bhv_device)
            ).item()
            return (
                torch.distributions.LogNormal(mu, sigma)
                .sample(size)
                .to(self.bhv_device)
            )

        # Else, generate constant speeds
        else:
            return torch.full(size, m, device=self.bhv_device)

    def _generate_random_head_dirs(self, mv_dirs: torch.Tensor, B: int, T: int):
        """Generate head directions for entire batch upfront."""
        ndim = self.gym.arena.ndim
        scale = self.cfg.look_around_scale

        if ndim == 2:
            # Existing 2D angle-based approach
            if scale > 0:
                head_variations = torch.randn((B, T, 1), device=self.bhv_device) * scale
                head_variations = torch.nn.functional.conv1d(
                    head_variations.transpose(1, 2),
                    torch.ones(1, 1, 5, device=self.bhv_device) / 5,
                    padding=2,
                ).transpose(1, 2)
            else:
                head_variations = torch.zeros((B, T, 1), device=self.bhv_device)

            mv_angles = torch.atan2(mv_dirs[:, :, 1], mv_dirs[:, :, 0])
            hd_angles = mv_angles + head_variations.squeeze(-1)
            return torch.stack([torch.cos(hd_angles), torch.sin(hd_angles)], dim=-1)

        else:
            # 3D: head direction as unit vector derived from mv_dir
            if scale > 0:
                # Random perturbation: add noise to mv_dir then renormalize
                noise = torch.randn((B, T, ndim), device=self.bhv_device) * scale
                # Temporal smoothing per component
                kernel = torch.ones(1, 1, 5, device=self.bhv_device) / 5
                for d in range(ndim):
                    noise[:, :, d:d+1] = torch.nn.functional.conv1d(
                        noise[:, :, d:d+1].transpose(1, 2),
                        kernel,
                        padding=2,
                    ).transpose(1, 2)
                hd = mv_dirs + noise
                hd = hd / hd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                return hd
            else:
                return mv_dirs.clone()

    def _recompute_maps(self):
        """Precompute boundary avoidance maps."""
        if self.cfg.avoid_boundary_dist <= 0:
            return
        if self.gym.arena is None or self.gym.arena.invmap_ is None:
            return

        invmap = self.gym.arena.invmap_.detach().cpu().numpy().astype(np.uint8)
        raw_dist = distance_transform_edt(invmap).astype(np.float32)

        # Gaussian decay: 1 at wall, 0 far away
        dist_map = np.exp(-(raw_dist ** 2 / self.cfg.avoid_boundary_dist))

        # Gradient of distance field → normal vectors pointing away from walls
        gradients = np.gradient(raw_dist)  # Returns ndim arrays
        normal_map = np.stack(gradients, axis=-1)  # (H, W, ndim) or (D, H, W, ndim)
        norms = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normal_map = normal_map / norms

        self.distance_map = torch.from_numpy(dist_map).float().to(self.bhv_device)
        self.wall_normal_map = torch.from_numpy(normal_map).float().to(self.bhv_device)
        self._boundary_dims = dist_map.shape
