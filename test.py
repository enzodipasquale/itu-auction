import torch

# Inputs
bidder_id = torch.tensor([0, 1, 2, 3, 10])
j_i       = torch.tensor([0, 1, 0, 10, 1])
bid_i     = torch.tensor([1.2, 2.5, 1.2, 0.9, 2.5])


unique_items, inverse = j_i.unique(return_inverse=True)

max_bid = torch.empty((unique_items.size(0),), dtype=bid_i.dtype)
max_bid.scatter_reduce_(0, inverse, bid_i, reduce='amax', include_self=False)

is_max = bid_i == max_bid[inverse]
winner = torch.empty((unique_items.size(0),),  dtype=bidder_id.dtype)
winner[inverse[is_max]] = bidder_id[is_max]


print("unique_items", unique_items)
print("winner", winner)
print("max_bid", max_bid)
