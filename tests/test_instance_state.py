import torch
import pytest

from batched_itu_auction import ITUauction as BatchedITUAuction
from itu_auction import get_template


def make_batched_tu(phi):
    def get_u(v, t, i, j=None):
        if j is None:
            return phi[t, i] - v[t]
        return phi[t, i, j] - v

    def get_v(u, t, j, i=None):
        if i is None:
            if t.ndim > 1:
                t = t.reshape(-1)
                j = j.reshape(-1)
                return phi[t][:, :, j] - u[t].unsqueeze(2)
            return phi[t, :, j] - u[t]
        return phi[t, i, j] - u

    return BatchedITUAuction(
        phi.shape[1],
        phi.shape[2],
        phi.shape[0],
        get_u,
        get_v,
        device=phi.device,
    )


def test_single_auctions_keep_their_valuations():
    first = get_template("TU")(torch.tensor([[4.0, 1.0], [2.0, 3.0]]))
    second = get_template("TU")(torch.tensor([[40.0, 10.0], [20.0, 30.0]]))

    first_u, _, _ = first.forward_auction(eps=0.1)
    second_u, _, _ = second.forward_auction(eps=0.1)

    assert first_u.max() < second_u.min()


def test_batched_auctions_keep_their_valuations():
    first = make_batched_tu(torch.tensor([[[4.0, 1.0], [2.0, 3.0]]]))
    second = make_batched_tu(torch.tensor([[[40.0, 10.0], [20.0, 30.0]]]))

    first_u, _, _, _ = first.forward_auction(eps=0.1)
    second_u, _, _, _ = second.forward_auction(eps=0.1)

    assert first_u.max() < second_u.min()


def test_batched_reverse_and_scaling_shapes():
    phi = torch.tensor(
        [
            [[4.0, 1.0], [2.0, 3.0], [1.0, 2.0]],
            [[3.0, 2.0], [1.0, 4.0], [2.0, 1.0]],
        ]
    )
    auction = make_batched_tu(phi)

    u, v, matching = auction.reverse_auction(eps=0.125, return_mu_t_i_j=True)

    assert u.shape == (2, 3)
    assert v.shape == (2, 2)
    assert matching.shape == (2, 3, 2)

    u, v, matching = auction.forward_reverse_scaling(2.0, 0.125, 0.5)

    assert u.shape == (2, 3)
    assert v.shape == (2, 2)
    assert matching.shape == (2, 3, 2)


@pytest.mark.parametrize("method", ["GS", "batched"])
def test_batched_reverse_selection_methods(method):
    phi = torch.tensor(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[3.0, 2.0], [1.0, 4.0]],
        ]
    )
    auction = make_batched_tu(phi)
    auction.method = method
    auction.sampling_rate = 0.5

    u, v, matching = auction.reverse_auction(eps=0.125, return_mu_t_i_j=True)

    assert u.shape == (2, 2)
    assert v.shape == (2, 2)
    assert torch.all(matching.sum(dim=1) <= 1)
    assert torch.all(matching.sum(dim=2) <= 1)
