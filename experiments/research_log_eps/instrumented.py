"""Instrumented version of ITUauction that counts iterations."""
import torch
from itu_auction.core import ITUauction as _ITUauction


class CountingAuction(_ITUauction):
    """Wraps ITUauction to count forward/reverse iterations per phase."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_counters()

    def _reset_counters(self):
        self.fwd_iters_per_phase = []
        self.rev_iters_per_phase = []
        self._cur_fwd = 0
        self._cur_rev = 0

    def _forward_iteration(self, *args, **kwargs):
        self._cur_fwd += 1
        return super()._forward_iteration(*args, **kwargs)

    def _reverse_iteration(self, *args, **kwargs):
        self._cur_rev += 1
        return super()._reverse_iteration(*args, **kwargs)

    def forward_auction(self, *args, **kwargs):
        self._cur_fwd = 0
        out = super().forward_auction(*args, **kwargs)
        self.fwd_iters_per_phase.append(self._cur_fwd)
        return out

    def reverse_auction(self, *args, **kwargs):
        self._cur_rev = 0
        out = super().reverse_auction(*args, **kwargs)
        self.rev_iters_per_phase.append(self._cur_rev)
        return out

    @property
    def total_iters(self):
        return sum(self.fwd_iters_per_phase) + sum(self.rev_iters_per_phase)

    @property
    def num_phases(self):
        return len(self.fwd_iters_per_phase) + len(self.rev_iters_per_phase)


def make_TU(num_i, num_j, seed=0, scale=10.0):
    """Build a TU instance by replacing the template."""
    torch.manual_seed(seed)
    Phi = torch.rand(num_i, num_j) * scale

    def get_U_i_j(v_j, i_id, j_id=None):
        if j_id is None:
            return Phi[i_id] - v_j.unsqueeze(0)
        return Phi[i_id, j_id] - v_j

    def get_V_i_j(u_i, j_id, i_id=None):
        if i_id is None:
            return Phi[:, j_id] - u_i.unsqueeze(1)
        return Phi[i_id, j_id] - u_i

    return CountingAuction(num_i, num_j, get_U_i_j, get_V_i_j), Phi


def make_LU(num_i, num_j, seed=0, scale=10.0, beta_min=1.0, beta_max=1.0):
    """Build a LU (linearly transferable) instance: U_ij(v) = Phi_ij - beta_ij * v."""
    torch.manual_seed(seed)
    Phi = torch.rand(num_i, num_j) * scale
    beta = torch.rand(num_i, num_j) * (beta_max - beta_min) + beta_min

    def get_U_i_j(v_j, i_id, j_id=None):
        if j_id is None:
            return Phi[i_id] - beta[i_id] * v_j.unsqueeze(0)
        return Phi[i_id, j_id] - beta[i_id, j_id] * v_j

    def get_V_i_j(u_i, j_id, i_id=None):
        if i_id is None:
            return (Phi[:, j_id] - u_i.unsqueeze(1)) / beta[:, j_id]
        return (Phi[i_id, j_id] - u_i) / beta[i_id, j_id]

    kappa = float(beta.max() / beta.min())
    return CountingAuction(num_i, num_j, get_U_i_j, get_V_i_j), Phi, beta, kappa
