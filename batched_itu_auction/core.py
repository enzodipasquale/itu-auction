import torch


def _assign_bids(group_ids, bidder_ids, bids, num_groups):
    active = torch.zeros(num_groups, dtype=torch.bool, device=bids.device)
    active[group_ids] = True
    active_ids = active.nonzero(as_tuple=True)[0]

    best_bids = torch.full(
        (num_groups,),
        float("-inf"),
        dtype=bids.dtype,
        device=bids.device,
    )
    best_bids.scatter_reduce_(
        0,
        group_ids,
        bids,
        reduce="amax",
        include_self=True,
    )
    is_best = bids == best_bids[group_ids]

    winners = torch.full(
        (num_groups,),
        torch.iinfo(torch.int32).max,
        dtype=torch.int32,
        device=bids.device,
    )
    winners.scatter_reduce_(
        0,
        group_ids[is_best],
        bidder_ids[is_best].to(torch.int32),
        reduce="amin",
        include_self=True,
    )
    return (
        active_ids,
        winners[active_ids].to(bidder_ids.dtype),
        best_bids[active_ids],
    )


class ITUauction:
    def __init__(
        self,
        num_i,
        num_j,
        num_t,
        get_U_t_i_j,
        get_V_t_i_j,
        lb=(0, 0),
        device=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.num_i = num_i
        self.num_j = num_j
        self.num_t = num_t

        self.all_i = torch.arange(num_i, device=self.device)
        self.all_j = torch.arange(num_j, device=self.device)
        self.all_t = torch.arange(num_t, device=self.device)

        self.u_0 = lb[0]
        self.v_0 = lb[1]

        self.init_v_t_j = torch.full(
            (self.num_t, self.num_j), self.v_0, dtype=torch.float, device=self.device
        )
        self.init_u_t_i = torch.full(
            (self.num_t, self.num_i), self.u_0, dtype=torch.float, device=self.device
        )
        self.init_mu_t_i = torch.full(
            (self.num_t, self.num_i), -1, dtype=torch.long, device=self.device
        )
        self.init_mu_t_j = torch.full(
            (self.num_t, self.num_j), -1, dtype=torch.long, device=self.device
        )

        self.method = None
        self.sampling_rate = None

        self.get_U_t_i_j = get_U_t_i_j
        self.get_V_t_i_j = get_V_t_i_j

    def check_equilibrium(self, u_t_i, v_t_j, mu_t_i_j, eps=0):
        cs = (
            self.get_U_t_i_j(
                v_t_j, self.all_t.unsqueeze(1), self.all_i.unsqueeze(0)
            ).amax(2)
            - u_t_i
        ).amax()
        feasible = torch.all(mu_t_i_j.sum(dim=2) <= 1) & torch.all(
            mu_t_i_j.sum(dim=1) <= 1
        )
        ir_i = torch.all(u_t_i[mu_t_i_j.sum(dim=2) == 0] <= self.u_0 + eps)
        ir_j = torch.all(v_t_j[mu_t_i_j.sum(dim=1) == 0] <= self.v_0 + eps)
        return cs, feasible, ir_i, ir_j

    def _forward_bid(self, unmatched_t_i, v_t_j, eps):
        t_id, i_id = unmatched_t_i

        U_t_i_j = self.get_U_t_i_j(v_t_j, t_id, i_id)

        top2 = U_t_i_j.topk(2, dim=-1)

        bidding = top2.values[:, 0] >= self.u_0 + eps

        bidder_t_i_id = bidder_t_id, bidder_i_id = t_id[bidding], i_id[bidding]
        out_t_i = t_id[~bidding], i_id[~bidding]

        j_ti = top2.indices[bidding, 0]
        w_ti = torch.clamp(top2.values[bidding, 1], min=self.u_0)

        bid_ti = self.get_V_t_i_j(w_ti - eps, bidder_t_id, j_ti, bidder_i_id)

        return out_t_i, bidder_t_i_id, j_ti, bid_ti

    def _forward_assign(self, bidder_t_i_id, j_ti, bid_ti):
        bidder_t_id, bidder_i_id = bidder_t_i_id
        group_ids = bidder_t_id * self.num_j + j_ti
        active_ids, winner_i_id, best_bid = _assign_bids(
            group_ids,
            bidder_i_id,
            bid_ti,
            self.num_t * self.num_j,
        )
        t_id = torch.div(active_ids, self.num_j, rounding_mode="floor")
        j_id = active_ids.remainder(self.num_j)
        return t_id, j_id, winner_i_id, best_bid

    def _forward_iteration(self, unmatched_t_i, v_t_j, mu_t_i, mu_t_j, eps):
        out_t_i, bidder_t_i_id, j_ti, bid_ti = self._forward_bid(
            unmatched_t_i, v_t_j, eps
        )
        mu_t_i[out_t_i] = self.num_j

        t_id, j_id, winner_i_id, best_bid = self._forward_assign(
            bidder_t_i_id, j_ti, bid_ti
        )

        reset_ti = mu_t_j[t_id, j_id]
        assigned_ti = reset_ti >= 0
        mu_t_i[t_id[assigned_ti], reset_ti[assigned_ti]] = -1
        mu_t_i[t_id, winner_i_id] = j_id
        mu_t_j[t_id, j_id] = winner_i_id

        v_t_j[t_id, j_id] = best_bid

        unmatched_t_i = (mu_t_i == -1).nonzero(as_tuple=True)

        return unmatched_t_i, v_t_j, mu_t_i, mu_t_j

    def _select_pairs(self, indices):
        if self.method == "GS":
            selected = slice(0, 1)
        elif self.method == "batched":
            if self.sampling_rate is None or not 0 < self.sampling_rate <= 1:
                raise ValueError("sampling_rate must be in (0, 1]")
            batch_size = max(1, int(self.sampling_rate * indices[0].numel()))
            selected = torch.randperm(indices[0].numel(), device=self.device)[
                :batch_size
            ]
        else:
            return indices
        return tuple(index[selected] for index in indices)

    def forward_auction(
        self,
        init_v_t_j=None,
        init_mu_t_i=None,
        init_mu_t_j=None,
        eps=0,
        return_mu_t_i_j=False,
    ):
        v_t_j = self.init_v_t_j.clone() if init_v_t_j is None else init_v_t_j
        mu_t_i = self.init_mu_t_i.clone() if init_mu_t_i is None else init_mu_t_i
        mu_t_j = self.init_mu_t_j.clone() if init_mu_t_j is None else init_mu_t_j
        unmatched_t_i = (mu_t_i == -1).nonzero(as_tuple=True)

        while unmatched_t_i[0].numel() > 0:
            unmatched_t_i = self._select_pairs(unmatched_t_i)
            unmatched_t_i, v_t_j, mu_t_i, mu_t_j = self._forward_iteration(
                unmatched_t_i, v_t_j, mu_t_i, mu_t_j, eps
            )

        u_t_i = (
            self.get_U_t_i_j(v_t_j, self.all_t.unsqueeze(1), self.all_i.unsqueeze(0))
            .amax(2)
            .clamp(min=self.u_0)
        )
        if return_mu_t_i_j:
            mu_t_i_j = mu_t_i[:, :, None] == self.all_j[None, None, :]
            return u_t_i, v_t_j, mu_t_i_j

        return u_t_i, v_t_j, mu_t_i, mu_t_j

    def _reverse_bid(self, unmatched_t_j, u_t_i, eps):
        t_id, j_id = unmatched_t_j

        V_t_i_j = self.get_V_t_i_j(u_t_i, t_id, j_id)
        top2 = V_t_i_j.topk(2, dim=-1)

        bidding = top2.values[:, 0] >= self.v_0 + eps
        bidder_t_j_id = bidder_t_id, bidder_j_id = t_id[bidding], j_id[bidding]
        out_t_i = t_id[~bidding], j_id[~bidding]

        i_tj = top2.indices[bidding, 0]
        w_tj = torch.clamp(top2.values[bidding, 1], min=self.v_0)

        bid_tj = self.get_U_t_i_j(w_tj, bidder_t_id, i_tj, bidder_j_id) + eps
        return out_t_i, bidder_t_j_id, i_tj, bid_tj

    def _reverse_assign(self, bidder_t_j_id, i_tj, bid_tj):
        bidder_t_id, bidder_j_id = bidder_t_j_id
        group_ids = bidder_t_id * self.num_i + i_tj
        active_ids, winner_j_id, best_bid = _assign_bids(
            group_ids,
            bidder_j_id,
            bid_tj,
            self.num_t * self.num_i,
        )
        t_id = torch.div(active_ids, self.num_i, rounding_mode="floor")
        i_id = active_ids.remainder(self.num_i)
        return t_id, i_id, winner_j_id, best_bid

    def _reverse_iteration(self, unmatched_t_j, u_t_i, mu_t_i, mu_t_j, eps):
        out_t_i, bidder_t_j_id, i_tj, bid_tj = self._reverse_bid(
            unmatched_t_j, u_t_i, eps
        )
        mu_t_j[out_t_i] = self.num_i

        t_id, i_id, winner_j_id, best_bid = self._reverse_assign(
            bidder_t_j_id, i_tj, bid_tj
        )

        reset_tj = mu_t_i[t_id, i_id]
        assigned_tj = reset_tj >= 0
        mu_t_j[t_id[assigned_tj], reset_tj[assigned_tj]] = -1
        mu_t_j[t_id, winner_j_id] = i_id
        mu_t_i[t_id, i_id] = winner_j_id

        u_t_i[t_id, i_id] = best_bid

        unmatched_t_j = (mu_t_j == -1).nonzero(as_tuple=True)
        return unmatched_t_j, u_t_i, mu_t_i, mu_t_j

    def reverse_auction(
        self,
        init_u_t_i=None,
        init_mu_t_i=None,
        init_mu_t_j=None,
        eps=0,
        return_mu_t_i_j=False,
    ):
        u_t_i = self.init_u_t_i.clone() if init_u_t_i is None else init_u_t_i
        mu_t_i = self.init_mu_t_i.clone() if init_mu_t_i is None else init_mu_t_i
        mu_t_j = self.init_mu_t_j.clone() if init_mu_t_j is None else init_mu_t_j
        unmatched_t_j = (mu_t_j == -1).nonzero(as_tuple=True)

        while unmatched_t_j[0].numel() > 0:
            unmatched_t_j = self._select_pairs(unmatched_t_j)
            unmatched_t_j, u_t_i, mu_t_i, mu_t_j = self._reverse_iteration(
                unmatched_t_j, u_t_i, mu_t_i, mu_t_j, eps
            )

        v_t_j = (
            self.get_V_t_i_j(u_t_i, self.all_t.unsqueeze(1), self.all_j.unsqueeze(0))
            .amax(dim=1)
            .clamp(min=self.v_0)
        )

        if return_mu_t_i_j:
            mu_t_i_j = mu_t_j[:, None, :] == self.all_i[None, :, None]
            return u_t_i, v_t_j, mu_t_i_j

        return u_t_i, v_t_j, mu_t_j

    def forward_reverse_scaling(self, eps_init, eps_target, scaling_factor):
        if eps_init <= 0 or eps_target <= 0:
            raise ValueError("epsilon values must be positive")
        if eps_init < eps_target:
            raise ValueError("eps_init must be at least eps_target")
        if not 0 < scaling_factor < 1:
            raise ValueError("scaling_factor must be between zero and one")

        eps = float(eps_init)
        v_t_j = self.init_v_t_j.clone()

        while True:
            u_t_i, v_t_j, mu_t_i, mu_t_j = self.forward_auction(
                init_v_t_j=v_t_j, eps=eps
            )
            eps = max(eps * scaling_factor, float(eps_target))
            u_t_i, v_t_j, mu_t_j = self.reverse_auction(init_u_t_i=u_t_i, eps=eps)
            if eps == eps_target:
                break

        u_t_i, v_t_j, mu_t_i_j = self.reverse_auction(
            init_u_t_i=u_t_i,
            eps=eps,
            return_mu_t_i_j=True,
        )
        mu_t_i = torch.where(mu_t_i_j.any(dim=2), mu_t_i_j.int().argmax(dim=2), -1)
        mu_t_j = torch.full(
            (self.num_t, self.num_j),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        valid = mu_t_i >= 0
        t_idx, i_idx = valid.nonzero(as_tuple=True)
        j_idx = mu_t_i[t_idx, i_idx]
        mu_t_j[t_idx, j_idx] = i_idx

        u_t_i, v_t_j, mu_t_i_j = self.forward_auction(
            init_v_t_j=v_t_j,
            init_mu_t_i=mu_t_i,
            init_mu_t_j=mu_t_j,
            eps=eps,
            return_mu_t_i_j=True,
        )

        return u_t_i, v_t_j, mu_t_i_j
