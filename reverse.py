# Reverse auction methods

def _reverse_bid(self, j_id, u_i, eps):
    # Compute top 2 values and indices at current prices
    V_i_j = self.get_V_i_j(u_i, j_id)
    top2 = V_i_j.topk(2, dim=1)

    # Filter out items preferring the outside option
    bidding = top2.values[:, 0] > self.lb_j + eps
    bidder_id, out_id = j_id[bidding], j_id[~bidding]

    # Compute selected agent and second best value for each item
    i_j = top2.indices[bidding, 0]
    w_j = top2.values[bidding, 1]

    # Compute bids
    bid_j = self.get_U_i_j(w_j - eps, i_j, bidder_id)

    return out_id, bidder_id, i_j, bid_j


def _reverse_assign(self, bidder_id, i_j, bid_j):
    unique_id, inverse = i_j.unique(return_inverse=True)

    best_bid = torch.empty(len(unique_id), dtype=bid_j.dtype, device=self.device)
    best_bid.scatter_reduce_(0, inverse, bid_j, reduce='amax', include_self=False)

    is_best = bid_j == best_bid[inverse]
    winner = torch.empty(len(unique_id), dtype=bidder_id.dtype, device=self.device)
    winner[inverse[is_best]] = bidder_id[is_best]

    return unique_id, winner, best_bid


def _reverse_iteration(self, unmatched_j, u_i, mu_j, eps):
    out_id, bidder_id, i_j, bid_j = self._reverse_bid(unmatched_j, u_i, eps)
    mu_j[out_id] = self.num_i

    unique_id, winner, best_bid = self._reverse_assign(bidder_id, i_j, bid_j)

    mask = torch.isin(mu_j, unique_id)
    mu_j[mask] = -1
    mu_j[winner] = unique_id

    u_i[unique_id] = best_bid
    unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]

    return unmatched_j, u_i, mu_j


def reverse_auction(self, init_u_i=None, init_mu_j=None, eps=0):
    u_i = torch.full((self.num_i,), self.lb_i, dtype=torch.float, device=self.device) if init_u_i is None else init_u_i
    mu_j = torch.full((self.num_j,), -1, dtype=torch.long, device=self.device) if init_mu_j is None else init_mu_j
    unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]

    while len(unmatched_j) > 0:
        unmatched_j, u_i, mu_j = self._reverse_iteration(unmatched_j, u_i, mu_j, eps)

    v_j = self.get_V_i_j(u_i).amax(dim=0).clamp(min=self.lb_j)
    mu_i_j = mu_j.unsqueeze(1) == self.all_i.unsqueeze(0)

    return u_i, v_j, mu_i_j
