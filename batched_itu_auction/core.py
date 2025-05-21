import torch
import sys

class ITUauction:
    def __init__(self, num_i, num_j, num_t, get_U_t_i_j, get_V_t_i_j, lb = (0, 0)):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_i = num_i
        self.num_j = num_j
        self.num_t = num_t

        self.all_i = torch.arange(num_i, device=self.device)
        self.all_j = torch.arange(num_j, device=self.device)
        self.all_t = torch.arange(num_t, device=self.device)

        self.u_0 = lb[0]
        self.v_0 = lb[1]

        self.init_v_t_j = torch.full((self.num_t, self.num_j), self.v_0, dtype=torch.float, device=self.device)
        self.init_u_t_i = torch.full((self.num_t, self.num_i), self.u_0, dtype=torch.float, device=self.device)
        self.init_mu_t_i = torch.full((self.num_t, self.num_i), -1, dtype=torch.long, device=self.device)
        self.init_mu_t_j = torch.full((self.num_t, self.num_j), -1, dtype=torch.long, device=self.device)

        self.method = None
        self.sampling_rate = None
    
        ITUauction.get_U_t_i_j = staticmethod(get_U_t_i_j)
        ITUauction.get_V_t_i_j = staticmethod(get_V_t_i_j)


    def check_equilibrium(self, u_t_i, v_t_j, mu_t_i_j, eps = 0):
        CS = (self.get_U_t_i_j(v_t_j, self.all_t.unsqueeze(1), self.all_i.unsqueeze(0)).amax(2) - u_t_i).amax()
        feas = torch.all((mu_t_i_j.sum(dim=2) <= 1)) and torch.all((mu_t_i_j.sum(dim=1) <= 1)) 
        IR_i =  torch.all(u_t_i[mu_t_i_j.sum(dim=2) == 0] <= self.u_0 + eps)
        IR_j =  torch.all(v_t_j[mu_t_i_j.sum(dim=1) == 0] <= self.v_0 + eps)

        print("=== Equilibrium Conditions ===")
        print(f"Complementary Slackness      : {CS:.4f}")
        print(f"Feasibility                  : {feas}")
        print(f"Individual Rationality (i)   : {IR_i}")
        print(f"Individual Rationality (j)   : {IR_j}")

        satisfied = CS <= eps and feas and IR_i and IR_j

        # if not satisfied:
        #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        #     print("=== Equilibrium Conditions Failed ===")
        #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        return CS, feas, IR_i, IR_j


    # Forward auction
    def _forward_bid(self, unmatched_t_i, v_t_j, eps):
        t_id, i_id = unmatched_t_i
        
        # Compute top 2 values and indices at current prices
        U_t_i_j = self.get_U_t_i_j(v_t_j, t_id, i_id)
        # print(U_t_i_j.shape)
        top2 = U_t_i_j.topk(2, dim=-1) 

        # Filter out bidders preferring the outside option
        bidding = top2.values[:, 0] >= self.u_0 + eps
        # print(top2.values.shape, bidding.shape)
        # sys.exit()
        bidder_t_i_id = bidder_t_id, bidder_i_id = t_id[bidding], i_id[bidding]
        out_t_i = t_id[~bidding], i_id[~bidding]

        # Compute selected item and second best value for each bidder
        j_ti = top2.indices[bidding, 0]
        w_ti = torch.clamp(top2.values[bidding, 1], min=self.u_0)

        # Compute bids
        bid_ti = self.get_V_t_i_j(w_ti - eps, bidder_t_id, j_ti, bidder_i_id)

        return out_t_i, bidder_t_i_id, j_ti, bid_ti

    def _forward_assign(self, bidder_t_i_id, j_ti, bid_ti):  
        bidder_t_id, bidder_i_id = bidder_t_i_id
        tj_pairs = torch.stack((bidder_t_id, j_ti), dim=1)
        unique_tj, inverse_indices = torch.unique(tj_pairs, dim=0, return_inverse=True)
        t_id = unique_tj[:, 0]
        j_id = unique_tj[:, 1]

        best_bid = torch.full((j_id.size(0),), float('-inf'), device=self.device)
        best_bid.scatter_reduce_(0, inverse_indices, bid_ti, reduce='amax', include_self=False)

        is_best = (bid_ti == best_bid[inverse_indices])
        winner_i_id = torch.empty(best_bid.numel(), dtype=bidder_i_id.dtype, device=self.device)
        winner_i_id[inverse_indices[is_best]] = bidder_i_id[is_best]

        return t_id, j_id, winner_i_id, best_bid

    def _forward_iteration(self, unmatched_t_i, v_t_j, mu_t_i, mu_t_j, eps):

        # Bidding phase
        out_t_i, bidder_t_i_id, j_ti, bid_ti = self._forward_bid(unmatched_t_i, v_t_j, eps)
        mu_t_i[out_t_i] = self.num_j
        
        # Assignment phase
        t_id, j_id, winner_i_id, best_bid = self._forward_assign(bidder_t_i_id, j_ti, bid_ti)

        # Update assignment
        reset_ti = mu_t_j[t_id, j_id] 
        assigned_ti = reset_ti >= 0
        mu_t_i[t_id[assigned_ti], reset_ti[assigned_ti]] = -1
        mu_t_i[t_id, winner_i_id] = j_id
        mu_t_j[t_id, j_id] = winner_i_id

        # Update prices
        v_t_j[t_id, j_id] = best_bid

        # Update unmatched bidders
        unmatched_t_i = (mu_t_i == -1).nonzero(as_tuple=True)

        return unmatched_t_i, v_t_j, mu_t_i, mu_t_j

    def forward_auction(self, init_v_t_j= None, init_mu_t_i= None, init_mu_t_j = None, eps = 0, return_mu_t_i_j = False):

        # Initialize prices and partial assignment
        v_t_j = self.init_v_t_j.clone() if init_v_t_j is None else init_v_t_j
        mu_t_i = self.init_mu_t_i.clone() if init_mu_t_i is None else init_mu_t_i
        mu_t_j = self.init_mu_t_j.clone() if init_mu_t_j is None else init_mu_t_j
        unmatched_t_i = (mu_t_i == -1).nonzero(as_tuple=True)
        
        # Iterate until all bidders are matched
        while unmatched_t_i[0].numel() > 0:
            unmatched_t_i, v_t_j, mu_t_i, mu_t_j = self._forward_iteration(unmatched_t_i, v_t_j, mu_t_i, mu_t_j, eps)

        # Compute utility for each bidder and binary assignment matrix
        u_t_i = self.get_U_t_i_j(v_t_j, self.all_t.unsqueeze(1), self.all_i.unsqueeze(0)).amax(2).clamp(min=self.u_0)
        if return_mu_t_i_j:
            mu_t_i_j = mu_t_i[:,:,None] == self.all_j[None,None,:]
            return u_t_i, v_t_j, mu_t_i_j

        return u_t_i, v_t_j, mu_t_i, mu_t_j


   # Reverse auction methods
    def _reverse_bid(self, unmatched_t_j, u_t_i, eps):
        t_id, j_id = unmatched_t_j
        
        # Compute top 2 values and indices at current prices
        V_t_i_j = self.get_V_t_i_j(u_t_i, t_id, j_id)
        top2 = V_t_i_j.topk(2, dim=-1) 

        # Filter out bidders preferring the outside option
        bidding = top2.values[:, 0] >= self.v_0 + eps
        bidder_t_j_id = bidder_t_id, bidder_j_id = t_id[bidding], j_id[bidding]
        out_t_i = t_id[~bidding], j_id[~bidding]
        
        # Compute selected item and second best value for each bidder
        i_tj = top2.indices[bidding, 0]
        w_tj = torch.clamp(top2.values[bidding, 1], min=self.v_0)

        # Compute bids
        bid_tj = self.get_U_t_i_j(w_tj , bidder_t_id, i_tj, bidder_j_id) + eps
        return out_t_i, bidder_t_j_id, i_tj, bid_tj

    def _reverse_assign(self, bidder_t_j_id, i_tj, bid_tj):
        bidder_t_id, bidder_j_id = bidder_t_j_id
        ti_pairs = torch.stack((bidder_t_id, i_tj), dim=1)
        unique_ti, inverse_indices = torch.unique(ti_pairs, dim=0, return_inverse=True)
        t_id = unique_ti[:, 0]
        i_id = unique_ti[:, 1]

        best_bid = torch.full((i_id.size(0),), float('-inf'), device=self.device)
        best_bid.scatter_reduce_(0, inverse_indices, bid_tj, reduce='amax', include_self=False)

        is_best = (bid_tj == best_bid[inverse_indices])
        winner_j_id = torch.empty(best_bid.numel(), dtype=bidder_j_id.dtype, device=self.device)
        winner_j_id[inverse_indices[is_best]] = bidder_j_id[is_best]

        return t_id, i_id, winner_j_id, best_bid

    def _reverse_iteration(self, unmatched_t_j, u_t_i, mu_t_i, mu_t_j, eps):
        # Bidding phase
        out_t_i, bidder_t_j_id, i_tj, bid_tj = self._reverse_bid(unmatched_t_j, u_t_i, eps)
        mu_t_j[out_t_i] = self.num_i

        # Assignment phase
        t_id, i_id, winner_j_id, best_bid = self._reverse_assign(bidder_t_j_id, i_tj, bid_tj)
        
        # Update assignment
        reset_tj = mu_t_i[t_id, i_id]
        assigned_tj = reset_tj >= 0
        mu_t_j[t_id[assigned_tj], reset_tj[assigned_tj]] = -1
        mu_t_j[t_id, winner_j_id] = i_id
        mu_t_i[t_id, i_id] = winner_j_id
        # Update prices
        u_t_i[t_id, i_id] = best_bid
        # Update unmatched bidders
        unmatched_t_j = (mu_t_j == -1).nonzero(as_tuple=True)
        return unmatched_t_j, u_t_i, mu_t_i, mu_t_j
  
    def reverse_auction(self, init_u_t_i= None, init_mu_t_i= None, init_mu_t_j = None, eps = 0, return_mu_t_i_j = False):
        u_t_i = self.init_u_t_i.clone() if init_u_t_i is None else init_u_t_i
        mu_t_i = self.init_mu_t_i.clone() if init_mu_t_i is None else init_mu_t_i
        mu_t_j = self.init_mu_t_j.clone() if init_mu_t_j is None else init_mu_t_j
        unmatched_t_j = (mu_t_j == -1).nonzero(as_tuple=True)

        while unmatched_t_j[0].numel() > 0:
            unmatched_t_j, u_t_i, mu_t_i, mu_t_j = self._reverse_iteration(unmatched_t_j, u_t_i, mu_t_i, mu_t_j, eps)

        v_t_j = self.get_V_t_i_j(u_t_i, self.all_t.unsqueeze(1), self.all_j.unsqueeze(0)).amax(2).clamp(min=self.v_0)
        if return_mu_t_i_j:
            mu_t_i_j = mu_t_i[:,:,None] == self.all_j[None,None,:]
            return u_t_i, v_t_j, mu_t_i_j

        return u_t_i, v_t_j, mu_t_i, mu_t_j

    # Scaling method
    def forward_reverse_scaling(self, eps_init, eps_target, scaling_factor):
        eps = eps_init
        v_t_j = self.init_v_t_j.clone()

        while True:
            print(f"eps: {eps}")
            u_t_i, v_t_j, mu_t_i, mu_t_j = self.forward_auction(init_v_t_j = v_t_j,  eps= eps)
            eps *=  scaling_factor
            u_t_i, v_t_j, mu_t_i, mu_t_j  = self.reverse_auction(init_u_t_i = u_t_i, eps= eps)
            if eps <= eps_target:
                break

        mu_t_j[mu_t_j == self.num_i] = -1
        violations_IR_t_i = (u_t_i > self.u_0) & (mu_t_i == -1)
        mu_t_i[violations_IR_t_i] = -1
        u_t_i, v_t_j, mu_t_i_j = self.forward_auction(init_v_t_j = v_t_j, init_mu_t_i= mu_t_i, init_mu_t_j = mu_t_j, eps= eps, return_mu_t_i_j= True)
        return u_t_i, v_t_j, mu_t_i_j



    # def forward_reverse_scaling(self, eps_init, eps_target, scaling_factor):
    #     eps = eps_init
    #     v_t_j = self.init_v_t_j.clone()
    #     mu_t_i = self.init_mu_t_i.clone()
    #     mu_t_j = self.init_mu_t_j.clone()

    #     while True:
    #         print(f"eps: {eps}")
    #         u_t_i, v_t_j, mu_t_i, mu_t_j = self.forward_auction(init_v_t_j = v_t_j, init_mu_t_i= mu_t_i, init_mu_t_j = mu_t_j, eps= eps)
    #         u_t_i, v_t_j, mu_t_i, mu_t_j = self.drop_for_scaling("j", u_t_i, v_t_j, mu_t_i, mu_t_j, eps)
    #         eps *=  scaling_factor
    #         u_t_i, v_t_j, mu_t_i, mu_t_j = self.reverse_auction(init_u_t_i = u_t_i, init_mu_t_i= mu_t_i, init_mu_t_j = mu_t_j, eps= eps)
    #         u_t_i, v_t_j, mu_t_i, mu_t_j = self.drop_for_scaling("i", u_t_i, v_t_j, mu_t_i, mu_t_j, eps)
    #         if eps <= eps_target:
    #             break

    #     u_t_i, v_t_j, mu_t_i_j = self.forward_auction(init_v_t_j = v_t_j, eps= eps, return_mu_t_i_j= True)
    #     return u_t_i, v_t_j, mu_t_i_j

    # def drop_for_scaling(self, side, u_t_i, v_t_j, mu_t_i, mu_t_j, eps):
    #     if side == "i":
    #         mu_t_j[mu_t_j == self.num_i] = -1
    #         violations_IR_t_i = (u_t_i > self.u_0) & (mu_t_i == -1)
    #         mu_t_i[violations_IR_t_i] = -1

            
    #     elif side == "j":
    #         mu_t_i[mu_t_i == self.num_j] = -1
    #         violations_IR_t_j = (v_t_j > self.v_0) & (mu_t_j == -1)
    #         mu_t_j[violations_IR_t_j] = -1

    #     return u_t_i, v_t_j, mu_t_i, mu_t_j
