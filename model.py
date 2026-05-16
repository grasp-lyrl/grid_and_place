import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNNCell(nn.Module):
    def __init__(
        self, hidden_size: int, alpha: float,
        ei_ratio: float = 0.8, homeostasis_eta: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ei_ratio = ei_ratio
        self.homeostasis_eta = homeostasis_eta

        self._init_ei()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.register_buffer("target_col_budget", torch.ones(hidden_size))
        self.reset_parameters()

    def _init_ei(self):
        self.n_exc = int(self.hidden_size * self.ei_ratio)
        self.n_inh = self.hidden_size - self.n_exc

        ei_sign = torch.ones(self.hidden_size)
        ei_sign[self.n_exc:] = -1

        idx = torch.randperm(self.hidden_size)
        ei_sign = ei_sign[idx]

        self.register_buffer("ei_sign", ei_sign)

    def reset_parameters(self):
        H = self.hidden_size
        with torch.no_grad():
            W = torch.randn(H, H) / math.sqrt(H)

            exc_mask = self.ei_sign > 0
            inh_mask = ~exc_mask

            # Dale at init
            W[:, exc_mask] = W[:, exc_mask].abs()
            W[:, inh_mask] = -W[:, inh_mask].abs()

            # set spectral radius on the effective matrix
            rho = torch.linalg.eigvals(W).abs().max().real
            W = W / (rho + 1e-8)

            # store magnitudes only; Dale sign is applied later
            W_mag = W.abs()
            self.weight_hh.copy_(W_mag)

            # preserve EXACT initial outgoing column budgets
            self.target_col_budget.copy_(W_mag.sum(dim=0).clamp_min(1e-8))

            W_eff = self._effective_weight()
            rho_eff = torch.linalg.eigvals(W_eff).abs().max().real.item()
            print(
                f"[E/I init] N_E={self.n_exc}, N_I={self.n_inh}, "
                f"init spectral radius={rho_eff:.3f}"
            )

    def _effective_weight(self):
        return torch.abs(self.weight_hh) * self.ei_sign

    @torch.no_grad()
    def project_ei_homeostasis_(self):
        """
        Column-wise synaptic scaling after optimizer.step().
        Uses self.homeostasis_eta: 1.0 => hard projection, <1.0 => softer, 0 => no-op.
        """
        if self.homeostasis_eta == 0:
            return
        W_mag = self.weight_hh.abs()
        col_sum = W_mag.sum(dim=0).clamp_min(1e-8)
        scale = (self.target_col_budget / col_sum).pow(self.homeostasis_eta)
        W_mag = W_mag * scale.unsqueeze(0)

        self.weight_hh.copy_(W_mag)

    def forward(self, cur_u, prev_h):
        W = self._effective_weight()

        pre_act = cur_u + F.linear(prev_h, W)
        h = self.alpha * F.softplus(pre_act) + (1.0 - self.alpha) * prev_h

        return h

class RNN(nn.Module):
    def __init__(self, d_clamped: int, d_free: int, input_dim: int, alpha: int, 
                 noise_level: float, homeostasis_eta: float, motion_dim: int):
        super().__init__()
        self.d_clamped = d_clamped
        self.d_free = d_free
        self.input_dim = input_dim
        self.d_model = d_clamped + d_free

        self._init_layout()
        self.motion_dim = motion_dim
        proj_input_dim = self.input_dim + motion_dim if motion_dim > 0 else self.input_dim
        self.input_proj = nn.Linear(proj_input_dim, self.d_clamped)
        self.output_proj = nn.Linear(self.d_clamped, self.input_dim)

        self.noise_level = noise_level
        self.rnn_cell = VanillaRNNCell(self.d_model, alpha, homeostasis_eta=homeostasis_eta)

    def _init_layout(self):
        c = self.d_clamped
        self.clamped_slice = slice(0, c)
        self.free_slice = slice(c, self.d_model)

        self._region_ranges = {
            "clamped": list(range(0, c)),
            "free": list(range(c, self.d_model)),
            "all": list(range(0, self.d_model)),
        }

    def _build_input(self, s_seq, m_seq=None):
        B, T = s_seq.shape[:2]

        proj_input = (
            torch.cat([s_seq, m_seq], dim=-1)
            if self.motion_dim > 0
            else s_seq
        )

        x = s_seq.new_zeros(B, T, self.d_model)
        x[..., self.clamped_slice] = self.input_proj(proj_input)

        return x

    def _apply_noise(self, h):
        if self.noise_level <= 0:
            return h
        std = torch.sqrt(h * self.noise_level + 1e-8)
        return F.relu(h + std * torch.randn_like(h))

    def _init_hidden(self, s_seq):
        B = s_seq.shape[0]
        return torch.zeros(B, self.d_model, device=s_seq.device, dtype=s_seq.dtype)

    def forward(self, s_seq, m_seq, h_init):
        h_init = h_init if h_init is not None else self._init_hidden(s_seq)
        input_seq = self._build_input(s_seq, m_seq)  # (B, T, D)

        B, T, _ = input_seq.shape
        hidden_seq = torch.empty(
            B, T, self.d_model,
            device=input_seq.device,
            dtype=input_seq.dtype,
        )

        h_tilde = h_init
        for t, x_t in enumerate(input_seq.unbind(1)):
            h_det = self.rnn_cell(x_t, h_tilde)
            h_tilde = self._apply_noise(h_det)
            hidden_seq[:, t] = h_tilde
        output_seq = self.output_proj(hidden_seq[:, :, self.clamped_slice])
        return hidden_seq, output_seq
