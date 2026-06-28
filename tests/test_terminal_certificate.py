import torch

from itu_auction.core import ITUauction


def make_tu_auction(phi):
    def get_U_i_j(v_j, i_id, j_id=None):
        if j_id is None:
            return phi[i_id] - v_j.unsqueeze(0)
        return phi[i_id, j_id] - v_j

    def get_V_i_j(u_i, j_id, i_id=None):
        if i_id is None:
            return phi[:, j_id] - u_i.unsqueeze(1)
        return phi[i_id, j_id] - u_i

    return ITUauction(phi.shape[0], phi.shape[1], get_U_i_j, get_V_i_j)


def test_terminal_certificate_rejects_unmatched_high_price():
    phi = torch.tensor([[20.0, 12.0], [0.0, 0.0]])
    auction = make_tu_auction(phi)
    v_j = torch.tensor([10.0, 3.0])
    mu_i_j = torch.tensor([[False, True], [False, False]])

    assert not auction.terminal_completion_certificate(v_j, mu_i_j, eps=1.0)


def test_terminal_certificate_accepts_outside_completed_items():
    phi = torch.tensor([[20.0, 12.0], [0.0, 0.0]])
    auction = make_tu_auction(phi)
    v_j = torch.tensor([0.0, 3.0])
    mu_i_j = torch.tensor([[False, True], [False, False]])

    assert auction.terminal_completion_certificate(v_j, mu_i_j, eps=1.0)


def test_certified_scaling_returns_terminal_certificate():
    phi = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
    auction = make_tu_auction(phi)

    u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
        eps_init=5.0,
        eps_target=0.1,
        scaling_factor=0.5,
        certify_terminal=True,
    )

    assert auction.terminal_completion_certificate(v_j, mu_i_j, eps=0.1)
    CS, feas, IR_i, IR_j = auction.check_equilibrium(u_i, v_j, mu_i_j, eps=0.1)
    assert CS <= 0.1
    assert feas
    assert IR_i
    assert IR_j
