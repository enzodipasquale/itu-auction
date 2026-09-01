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


def make_ltu_auction(phi, beta):
    def get_U_i_j(v_j, i_id, j_id=None):
        if j_id is None:
            return phi[i_id] - beta[i_id] * v_j.unsqueeze(0)
        return phi[i_id, j_id] - beta[i_id, j_id] * v_j

    def get_V_i_j(u_i, j_id, i_id=None):
        if i_id is None:
            return (phi[:, j_id] - u_i.unsqueeze(1)) / beta[:, j_id]
        return (phi[i_id, j_id] - u_i) / beta[i_id, j_id]

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


def test_reverse_bid_uses_inverse_frontier_for_nonunit_slope():
    phi = torch.tensor([[10.0], [6.0]])
    beta = torch.tensor([[2.0], [3.0]])
    auction = make_ltu_auction(phi, beta)

    _, bidder_id, i_j, bid_j = auction._reverse_bid(
        torch.tensor([0]), torch.tensor([0.0, 0.0]), eps=0.5
    )

    assert bidder_id.tolist() == [0]
    assert i_j.tolist() == [0]
    implied_v = auction.get_V_i_j(bid_j, bidder_id, i_j)
    assert torch.allclose(implied_v, torch.tensor([1.5]))


def test_matching_preserve_terminal_releases_violated_matched_consumer():
    phi = torch.tensor([[20.0, 12.0], [0.0, 0.0]])
    auction = make_tu_auction(phi)

    u_i, v_j, mu_i = auction.matching_preserve_forward_auction(
        init_v_j=torch.tensor([10.0, 0.0]),
        init_mu_i=torch.tensor([0, 1]),
        eps=1.0,
    )

    assert mu_i.tolist() == [1, auction.num_j]
    assert torch.allclose(u_i, torch.tensor([9.0, 0.0]))
    mu_i_j = mu_i.unsqueeze(1) == auction.all_j.unsqueeze(0)
    assert not auction.terminal_completion_certificate(v_j, mu_i_j, eps=1.0)


def test_forward_auction_returns_matched_equality_payoffs():
    phi = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
    auction = make_tu_auction(phi)

    u_i, v_j, mu_i_j = auction.forward_auction(eps=0.1, return_mu_i_j=True)

    matched_i, matched_j = mu_i_j.nonzero(as_tuple=True)
    assert torch.allclose(u_i[matched_i], phi[matched_i, matched_j] - v_j[matched_j])
    assert torch.allclose(u_i[~mu_i_j.any(dim=1)], torch.zeros_like(u_i[~mu_i_j.any(dim=1)]))


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
    assert auction.last_terminal_eps == 0.1
    CS, feas, IR_i, IR_j = auction.check_equilibrium(u_i, v_j, mu_i_j, eps=0.1)
    assert CS <= 0.1
    assert feas
    assert IR_i
    assert IR_j
