  # def forward_auction(self, init_v_j= None, init_mu_i= None, eps = 0):

    #     v_j = torch.full((self.num_j,), self.lb_j, dtype=torch.float, device=self.device) if init_v_j is None else init_v_j
    #     mu_i = torch.full((self.num_i,), -1, dtype=torch.long, device=self.device) if init_mu_i is None else init_mu_i

    #     unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0]

    #     while len(unmatched_i) > 0:
    #         # Bidding phase
    #         out_id, bidder_id, j_i, bid_i = self._forward_bid(unmatched_i, v_j, eps)
    #         mu_i[out_id] = self.num_j

    #         # Assignment phase
    #         unique_items, winner, best_bid = self._forward_assign(bidder_id, j_i, bid_i)

    #         # Update assignment
    #         mask = torch.isin(mu_i, unique_items)
    #         mu_i[mask] = -1 
    #         mu_i[winner] = unique_items

    #         # Update prices
    #         v_j[unique_items] = best_bid

    #         # Update unmatched bidders
    #         unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0]


    #     u_i = self.get_U_i_j(v_j).amax(dim=1).clamp(min=self.lb_i)
    #     mu_i_j = mu_i.unsqueeze(1) == self.all_j.unsqueeze(0)

    #     return u_i, v_j, mu_i_j