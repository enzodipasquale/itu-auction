from itu_auction.core import ITUauction
import torch

n_i = 5
m_j = 6
torch.manual_seed(0)
Φ_i_j = torch.rand(n_i, m_j)

def get_U_i_j(v_j, i_id, j_id=None):
    if j_id is None:
        return Φ_i_j[i_id] - v_j.unsqueeze(0)
    else:
        return Φ_i_j[i_id, j_id] - v_j

def get_V_i_j(u_i, j_id, i_id=None):
    if i_id is None:
        return Φ_i_j[:, j_id] - u_i.unsqueeze(0)
    else:
        # return Φ_i_j[i_id.unsqueeze(1), j_id.unsqueeze(0)] - u_i.unsqueeze(0)
        # gather
        # return Φ_i_j[:, j_id].gather(0, i_id.unsqueeze(1)) - u_i.unsqueeze(0) 
        return Φ_i_j[i_id, j_id] - u_i

TU_example = ITUauction(n_i, m_j, get_U_i_j, get_V_i_j)


# Example usage:


w_i = torch.rand(4)


# print(TU_example.get_U_i_j(v_j, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])))
i_id = torch.tensor([0, 1, 3, 4])
j_id = torch.tensor([0, 0, 1, 1])

# cond = w_i < -132443
# w_i = w_i[cond]
# i_id = i_id[cond]
# j_id = j_id[cond]


# print(Φ_i_j[i_id.unsqueeze(1), j_id.unsqueeze(0)])
# print(TU_example.get_V_i_j(w_i, i_id, j_id))

import torch

# bidder_id = torch.tensor([0, 1, 2, 3, 10])
j_i        = torch.tensor([0, 1, 0, 10, 1])
bid_i      = torch.tensor([1.2, 2.5, 3.1, 0.9, 2.5])

unique_items, inverse = j_i.unique(return_inverse=True)
max_bid = torch.full((unique_items.size(0),), float('-inf'))

print(max_bid)

max_bid.scatter_reduce_(0, inverse, bid_i, reduce='amax', include_self=True)

print(max_bid)
