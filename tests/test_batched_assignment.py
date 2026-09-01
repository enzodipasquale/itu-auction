import torch

from batched_itu_auction.core import _assign_bids


def test_assign_bids_uses_smallest_bidder_to_break_ties():
    group_ids = torch.tensor([3, 1, 3, 1])
    bidder_ids = torch.tensor([2, 4, 1, 0])
    bids = torch.tensor([5.0, 2.0, 5.0, 3.0])

    active, winners, best = _assign_bids(group_ids, bidder_ids, bids, 5)

    assert torch.equal(active, torch.tensor([1, 3]))
    assert torch.equal(winners, torch.tensor([0, 1]))
    assert torch.equal(best, torch.tensor([3.0, 5.0]))
