import torch

num_i = 5
num_j = 7
num_t = 2



torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Φ_t_i_j = torch.stack([
# #                             (torch.randint(0, 3, (num_i, 1), device=device) - torch.randint(0, 3, (1, num_j), device=device)) ** 2
# #                                 for _ in range(num_t)], dim=0)

Φ_i_j = (torch.randint(0, 3, (num_i, 1), device=device) - torch.randint(0, 3, (1, num_j), device=device)) ** 2
# row = torch.tensor([0,0,0])
# col = torch.tensor([0,0,2])

# Φ_i_j[row, col] = torch.tensor([5,0,2])
# print(Φ_i_j)

bidder_t_id, bidder_i_id = torch.tensor([0, 0, 0, 0, 0, 1, 1]), torch.tensor([0, 1, 2, 3, 4, 0, 1])
j_ti = torch.tensor([0, 5, 0, 5, 0, 0, 0])
bid_ti = torch.tensor([1, 0.5, 0.5, .5, 0.5, 1, 1])

bidder_t_id = torch.tensor([0, 0, 0, 0, 0, 1, 1])
j_ti = torch.tensor([0, 5, 0, 5, 0, 0, 0])



tj_pairs = torch.stack((bidder_t_id, j_ti), dim=1)
unique_tj, inverse_indices = torch.unique(tj_pairs, dim=0, return_inverse=True)
unique_t = unique_tj[:, 0]
unique_j = unique_tj[:, 1]

max_bids = torch.full((unique_j.size(0),), float('-inf'))
max_bids.scatter_reduce_(0, inverse_indices, bid_ti, reduce='amax', include_self=False)

is_best = (bid_ti == max_bids[inverse_indices])
winner_i_id = torch.empty(max_bids.numel(), dtype=bidder_i_id.dtype)
winner_i_id[inverse_indices[is_best]] = bidder_i_id[is_best]

print(unique_t)
print(unique_j)
print(max_bids)
print(winner_i_id)

