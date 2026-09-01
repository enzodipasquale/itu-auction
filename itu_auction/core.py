import torch


class ITUauction:
    def __init__(self, num_i, num_j, get_U_i_j, get_V_i_j, lb=(0, 0), device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.num_i = num_i
        self.num_j = num_j

        self.all_i = torch.arange(num_i, device=self.device)
        self.all_j = torch.arange(num_j, device=self.device)

        self.u_0 = lb[0]
        self.v_0 = lb[1]

        self.init_v_j = torch.full(
            (self.num_j,), self.v_0, dtype=torch.float, device=self.device
        )
        self.init_u_i = torch.full(
            (self.num_i,), self.u_0, dtype=torch.float, device=self.device
        )
        self.init_mu_i = torch.full(
            (self.num_i,), -1, dtype=torch.long, device=self.device
        )
        self.init_mu_j = torch.full(
            (self.num_j,), -1, dtype=torch.long, device=self.device
        )

        self.method = None
        self.sampling_rate = None

        self.get_U_i_j = get_U_i_j
        self.get_V_i_j = get_V_i_j

    def check_equilibrium(self, u_i, v_j, mu_i_j, eps=0):
        cs = (self.get_U_i_j(v_j, self.all_i).amax(dim=1) - u_i).amax()
        feasible = torch.all(mu_i_j.sum(dim=1) <= 1) & torch.all(
            (mu_i_j.sum(dim=0) <= 1)
        )
        ir_i = torch.all(u_i[mu_i_j.sum(dim=1) == 0] <= self.u_0 + eps)
        ir_j = torch.all(v_j[mu_i_j.sum(dim=0) == 0] <= self.v_0 + eps)
        return cs, feasible, ir_i, ir_j

    def terminal_completion_certificate(self, v_j, mu_i_j, eps, outside_tol=0.0):
        matched_j = mu_i_j.sum(dim=0) > 0
        lower_ok = torch.all(v_j >= self.v_0 - eps - outside_tol)

        if torch.all(matched_j):
            outside_ok = torch.tensor(True, device=v_j.device)
        else:
            outside_ok = torch.all(torch.abs(v_j[~matched_j] - self.v_0) <= outside_tol)

        return bool(lower_ok and outside_ok)

    def _mu_i_to_mu_i_j(self, mu_i):
        return mu_i.unsqueeze(1) == self.all_j.unsqueeze(0)

    def _mu_j_to_mu_i_j(self, mu_j):
        return mu_j.unsqueeze(0) == self.all_i.unsqueeze(1)

    def _mu_j_to_mu_i(self, mu_j):
        mu_i = self.init_mu_i.clone()
        real_j = (mu_j >= 0) & (mu_j < self.num_i)
        if torch.any(real_j):
            mu_i[mu_j[real_j]] = self.all_j[real_j]
        return mu_i

    def _consumer_payoffs_from_matching(self, v_j, mu_i):
        u_i = self.init_u_i.clone()
        real_i = (mu_i >= 0) & (mu_i < self.num_j)
        if torch.any(real_i):
            i_id = self.all_i[real_i]
            j_id = mu_i[real_i]
            u_i[real_i] = self.get_U_i_j(v_j[j_id], i_id, j_id)
        return u_i

    def _item_payoffs_from_matching(self, u_i, mu_j):
        v_j = self.init_v_j.clone()
        real_j = (mu_j >= 0) & (mu_j < self.num_i)
        if torch.any(real_j):
            j_id = self.all_j[real_j]
            i_id = mu_j[real_j]
            v_j[real_j] = self.get_V_i_j(u_i[i_id], j_id, i_id)
        return v_j

    def _forward_eligible_i(self, v_j, mu_i, eps):
        unmatched_i = mu_i == -1
        current_u = self._consumer_payoffs_from_matching(v_j, mu_i)
        best_real_u = self.get_U_i_j(v_j, self.all_i).amax(dim=1)
        outside_u = torch.full_like(best_real_u, self.u_0)
        best_u = torch.maximum(best_real_u, outside_u)
        violated_i = best_u > current_u + eps
        return (unmatched_i | violated_i).nonzero(as_tuple=True)[0]

    def _forward_bid(self, i_id, v_j, eps):
        U_i_j = self.get_U_i_j(v_j, i_id)
        top2 = U_i_j.topk(2, dim=1)

        bidding = top2.values[:, 0] >= self.u_0 + eps
        bidder_id, out_id = i_id[bidding], i_id[~bidding]

        j_i = top2.indices[bidding, 0]

        w_i = torch.clamp(top2.values[bidding, 1], min=self.u_0)

        bid_i = self.get_V_i_j(w_i - eps, j_i, bidder_id)

        return out_id, bidder_id, j_i, bid_i

    def _forward_assign(self, bidder_id, j_i, bid_i):
        unique_id, inverse = j_i.unique(return_inverse=True)

        best_bid = torch.empty(len(unique_id), dtype=bid_i.dtype, device=self.device)
        best_bid.scatter_reduce_(0, inverse, bid_i, reduce="amax", include_self=False)

        is_best = bid_i == best_bid[inverse]
        winner = torch.empty(len(unique_id), dtype=bidder_id.dtype, device=self.device)
        winner[inverse[is_best]] = bidder_id[is_best]

        return unique_id, winner, best_bid

    def _forward_iteration(self, unmatched_i, v_j, mu_i, eps):
        out_id, bidder_id, j_i, bid_i = self._forward_bid(unmatched_i, v_j, eps)
        mu_i[out_id] = self.num_j

        unique_id, winner, best_bid = self._forward_assign(bidder_id, j_i, bid_i)

        reset_i = torch.isin(mu_i, unique_id)

        mu_i[reset_i] = -1
        mu_i[winner] = unique_id

        v_j[unique_id] = best_bid

        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0]

        return unmatched_i, v_j, mu_i

    def _select_indices(self, indices):
        if self.method == "GS":
            return indices[:1]
        if self.method != "batched":
            return indices
        if self.sampling_rate is None or not 0 < self.sampling_rate <= 1:
            raise ValueError("sampling_rate must be in (0, 1]")
        batch_size = max(1, int(self.sampling_rate * indices.numel()))
        order = torch.randperm(indices.numel(), device=indices.device)
        return indices[order[:batch_size]]

    def forward_auction(
        self, init_v_j=None, init_mu_i=None, eps=0, return_mu_i_j=False
    ):
        v_j = self.init_v_j.clone() if init_v_j is None else init_v_j
        mu_i = self.init_mu_i.clone() if init_mu_i is None else init_mu_i
        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0]

        while unmatched_i.numel() > 0:
            unmatched_i = self._select_indices(unmatched_i)
            unmatched_i, v_j, mu_i = self._forward_iteration(
                unmatched_i, v_j, mu_i, eps
            )

        u_i = self._consumer_payoffs_from_matching(v_j, mu_i)
        if return_mu_i_j:
            mu_i_j = self._mu_i_to_mu_i_j(mu_i)
            return u_i, v_j, mu_i_j

        return u_i, v_j, mu_i

    def matching_preserve_forward_auction(
        self,
        init_v_j=None,
        init_mu_i=None,
        eps=0,
        return_mu_i_j=False,
    ):
        v_j = self.init_v_j.clone() if init_v_j is None else init_v_j
        mu_i = self.init_mu_i.clone() if init_mu_i is None else init_mu_i
        eligible_i = self._forward_eligible_i(v_j, mu_i, eps)

        while eligible_i.numel() > 0:
            selected_i = self._select_indices(eligible_i)
            mu_i[selected_i] = -1
            _, v_j, mu_i = self._forward_iteration(selected_i, v_j, mu_i, eps)
            eligible_i = self._forward_eligible_i(v_j, mu_i, eps)

        u_i = self._consumer_payoffs_from_matching(v_j, mu_i)
        if return_mu_i_j:
            return u_i, v_j, self._mu_i_to_mu_i_j(mu_i)
        return u_i, v_j, mu_i

    def _reverse_bid(self, j_id, u_i, eps):
        V_i_j = self.get_V_i_j(u_i, j_id)
        top2 = V_i_j.topk(2, dim=0)

        bidding = top2.values[0] >= self.v_0 + eps
        bidder_id, out_id = j_id[bidding], j_id[~bidding]

        i_j = top2.indices[0, bidding]

        w_j = torch.clamp(top2.values[1, bidding], min=self.v_0)

        bid_j = self.get_U_i_j(w_j - eps, i_j, bidder_id)

        return out_id, bidder_id, i_j, bid_j

    def _reverse_assign(self, bidder_id, i_j, bid_j):
        unique_id, inverse = i_j.unique(return_inverse=True)

        best_bid = torch.empty(len(unique_id), dtype=bid_j.dtype, device=self.device)
        best_bid.scatter_reduce_(0, inverse, bid_j, reduce="amax", include_self=False)

        is_best = bid_j == best_bid[inverse]
        winner = torch.empty(len(unique_id), dtype=bidder_id.dtype, device=self.device)
        winner[inverse[is_best]] = bidder_id[is_best]

        return unique_id, winner, best_bid

    def _reverse_iteration(self, unmatched_j, u_i, mu_j, eps):
        out_id, bidder_id, i_j, bid_j = self._reverse_bid(unmatched_j, u_i, eps)
        mu_j[out_id] = self.num_i

        unique_id, winner, best_bid = self._reverse_assign(bidder_id, i_j, bid_j)

        reset_j = torch.isin(mu_j, unique_id)
        mu_j[reset_j] = -1
        mu_j[winner] = unique_id

        u_i[unique_id] = best_bid
        unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]

        return unmatched_j, u_i, mu_j

    def reverse_auction(
        self, init_u_i=None, init_mu_j=None, eps=0, return_mu_i_j=False
    ):
        u_i = self.init_u_i.clone() if init_u_i is None else init_u_i
        mu_j = self.init_mu_j.clone() if init_mu_j is None else init_mu_j
        unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]

        while unmatched_j.numel() > 0:
            unmatched_j = self._select_indices(unmatched_j)
            unmatched_j, u_i, mu_j = self._reverse_iteration(
                unmatched_j, u_i, mu_j, eps
            )

        v_j = self._item_payoffs_from_matching(u_i, mu_j)

        if return_mu_i_j:
            mu_i_j = self._mu_j_to_mu_i_j(mu_j)
            return u_i, v_j, mu_i_j

        return u_i, v_j, mu_j

    def forward_reverse_scaling(
        self,
        eps_init,
        eps_target,
        scaling_factor,
        certify_terminal=False,
        certificate_tol=0.0,
    ):
        eps = eps_init
        v_j = self.init_v_j.clone()
        mu_i = self.init_mu_i.clone()

        while True:
            u_i, v_j, mu_i = self.forward_auction(init_v_j=v_j, eps=eps)
            next_eps = eps * scaling_factor
            if next_eps <= eps_target:
                break
            u_i, v_j, mu_j = self.reverse_auction(init_u_i=u_i, eps=next_eps)
            mu_i = self._mu_j_to_mu_i(mu_j)
            eps = next_eps * scaling_factor
            if eps <= eps_target:
                break

        terminal_eps = eps_target

        u_i, v_j, mu_i_j = self.matching_preserve_forward_auction(
            init_v_j=v_j,
            init_mu_i=mu_i,
            eps=terminal_eps,
            return_mu_i_j=True,
        )

        terminal_certificate_passed = self.terminal_completion_certificate(
            v_j, mu_i_j, eps=terminal_eps, outside_tol=certificate_tol
        )
        self.last_terminal_eps = terminal_eps
        self.last_terminal_certificate_passed = terminal_certificate_passed
        self.last_terminal_fallback_used = False

        if certify_terminal and not terminal_certificate_passed:
            self.last_terminal_fallback_used = True
            u_i, v_j, mu_i_j = self.forward_auction(
                init_v_j=self.init_v_j.clone(),
                eps=terminal_eps,
                return_mu_i_j=True,
            )

        self.last_terminal_final_certificate_passed = (
            self.terminal_completion_certificate(
                v_j, mu_i_j, eps=terminal_eps, outside_tol=certificate_tol
            )
        )

        return u_i, v_j, mu_i_j
