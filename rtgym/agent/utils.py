import torch
import hashlib
from dataclasses import dataclass
from rtgym.dataclasses import AgentState, Trajectory


@dataclass
class TrajectoryGeneratorConfig:
    spd_mean: float = 0.0
    spd_sd: float = 0.0
    alpha_spd: float = 0.1
    alpha_dir: float = 0.1
    switch_dir_prob: float = 0.0
    switch_spd_prob: float = 0.0
    avoid_boundary_dist: float = -1
    steer_gain: float = 1.0
    max_blend: float = 0.35
    proximity_sharpness: float = 6.0
    look_around_scale: float = 0.0

    @classmethod
    def from_profile(cls, profile: dict, gym=None):
        """Create config from profile with validation and auto-conversion"""
        config = cls()
        config.update_from_profile(profile, gym)
        return config

    def update_from_profile(self, profile: dict, gym=None):
        """Update config from profile with validation"""
        # Core movement parameters
        self.spd_mean = self._safe_float(profile, "spd_mean", required=True)
        self.spd_sd = self._safe_float(profile, "spd_sd", default=0.0)
        self.alpha_spd = self._safe_float(profile, "alpha_spd", default=0.1)
        self.alpha_dir = self._safe_float(profile, "alpha_dir", default=0.1)

        # Switching probabilities
        self.switch_dir_prob = self._safe_float(
            profile, "switch_dir_prob", required=True
        )
        self.switch_spd_prob = self._safe_float(
            profile, "switch_spd_prob", required=True
        )

        # Boundary avoidance - special handling
        boundary_avoidance = profile.get("boundary_avoidance", None)
        if boundary_avoidance is not None:
            self._setup_boundary_avoidance(float(boundary_avoidance), gym)
        else:
            self._disable_boundary_avoidance()

        # Head direction variation
        self.look_around_scale = self._safe_float(
            profile, "look_around_scale", default=0.0
        )

        # Validate all parameters
        self.validate()

    def _safe_float(self, profile: dict, key: str, default=None, required=False):
        """Safely convert profile value to float with validation"""
        if key not in profile:
            if required:
                raise ValueError(f"Required parameter '{key}' missing from profile")
            return default
        try:
            return float(profile[key])
        except (ValueError, TypeError):
            raise ValueError(
                f"Parameter '{key}' must be convertible to float, got {profile[key]}"
            )

    def _setup_boundary_avoidance(self, avoidance_level: float, gym):
        if gym is None:
            raise ValueError("gym required for boundary avoidance setup")

        # Convert to speed-invariant parameters
        typical_speed = self.spd_mean * gym.t_res / 1e3

        # Balanced parameters
        self.avoid_boundary_dist = max(
            12.0 * typical_speed * (0.5 + 0.5 * avoidance_level), 1e-6
        )
        self.steer_gain = 2.0 + 1.5 * avoidance_level
        self.proximity_sharpness = 3.0 + 2.0 * (1 - avoidance_level)
        self.max_blend = 0.3 + 0.2 * avoidance_level

    def _disable_boundary_avoidance(self):
        self.avoid_boundary_dist = -1
        self.steer_gain = 0.0
        self.proximity_sharpness = 1.0
        self.max_blend = 0.0

    def validate(self):
        validations = [
            (self.spd_mean > 0, "spd_mean must be positive, got {self.spd_mean}"),
            (self.spd_sd >= 0, "spd_sd must be non-negative, got {self.spd_sd}"),
            (
                0 <= self.alpha_spd <= 1,
                "alpha_spd must be in [0, 1], got {self.alpha_spd}",
            ),
            (
                0 <= self.alpha_dir <= 1,
                "alpha_dir must be in [0, 1], got {self.alpha_dir}",
            ),
            (
                0 <= self.switch_dir_prob <= 1,
                "switch_dir_prob must be in [0, 1], got {self.switch_dir_prob}",
            ),
            (
                0 <= self.switch_spd_prob <= 1,
                "switch_spd_prob must be in [0, 1], got {self.switch_spd_prob}",
            ),
            (
                0 <= self.look_around_scale <= 1,
                "look_around_scale must be in [0, 1], got {self.look_around_scale}",
            ),
        ]
        for valid, msg in validations:
            if not valid:
                raise ValueError(msg.format(self=self))


class TrajectoryBuilder:
    """Trajectory builder class"""

    def __init__(self, max_length: int, batch_size: int, ndim: int = 2, device: str = "cpu"):
        self.batch_size = batch_size
        self.max_length = max_length
        self.ndim = ndim
        self.cur_ts = 0
        self.device = torch.device(device)

        # Pre-allocate all tensors
        self.coord = torch.zeros(batch_size, max_length, ndim, device=self.device)
        self.spd = torch.zeros(batch_size, max_length, 1, device=self.device)
        self.mv_dir = torch.zeros(batch_size, max_length, ndim, device=self.device)
        self.head_dir = torch.zeros(batch_size, max_length, ndim, device=self.device)

    def append(self, agent_state: AgentState):
        """Append agent state to the trajectory"""
        self.set_state(self.cur_ts, agent_state)

    def set_state(self, ts: int, agent_state: AgentState):
        """Set agent state at timestep - optimized for batch operations"""
        self.coord[:, ts] = agent_state.coord
        self.spd[:, ts] = agent_state.spd
        self.mv_dir[:, ts] = agent_state.mv_dir
        self.head_dir[:, ts] = agent_state.head_dir
        self.cur_ts = max(self.cur_ts, ts + 1)

    def finalize(self) -> Trajectory:
        """Finalize by computing displacement and trimming to actual length"""
        # Trim to actual used length
        final_coord = self.coord[:, : self.cur_ts]
        final_spd = self.spd[:, : self.cur_ts]
        final_mv_dir = self.mv_dir[:, : self.cur_ts]
        final_head_dir = self.head_dir[:, : self.cur_ts]

        return Trajectory(
            coord=final_coord,
            head_dir=final_head_dir,
            spd=final_spd,
            mv_dir=final_mv_dir,
            device=self.device.type,
        )


def hash_seed(seed, sensory_key):
    str_hash = int(hashlib.sha256(sensory_key.encode("utf-8")).hexdigest(), 16) % (10**8)
    return seed + str_hash
