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
        # self.init_u_i = torch.full((self.num_i,), self.u_0 , dtype=torch.float, device=self.device)
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

        if not satisfied:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("=== Equilibrium Conditions Failed ===")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        return CS, feas, IR_i, IR_j


    # Forward auction
    def _forward_bid(self, unmatched_t_i, v_t_j, eps):
        t_id, i_id = unmatched_t_i
        
        # Compute top 2 values and indices at current prices
        U_t_i_j = self.get_U_t_i_j(v_t_j, t_id, i_id)
        top2 = U_t_i_j.topk(2, dim=-1) 

        # Filter out bidders preferring the outside option
        bidding = top2.values[:, 0] >= self.u_0 + eps

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

        best_bid = torch.full((j_id.size(0),), float('-inf'))
        best_bid.scatter_reduce_(0, inverse_indices, bid_ti, reduce='amax', include_self=False)

        is_best = (bid_ti == best_bid[inverse_indices])
        winner_i_id = torch.empty(best_bid.numel(), dtype=bidder_i_id.dtype)
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
            # if self.method == "GS":
            #     unmatched_i = unmatched_i[:1]
            # if self.method == "batched":
            #     batch_size = max(1, int(self.sampling_rate * unmatched_i.size(0)))
            #     batch = unmatched_i[torch.randperm(unmatched_i.size(0))[:batch_size]]
            
            unmatched_t_i, v_t_j, mu_t_i, mu_t_j = self._forward_iteration(unmatched_t_i, v_t_j, mu_t_i, mu_t_j, eps)

        # Compute utility for each bidder and binary assignment matrix
        u_t_i = self.get_U_t_i_j(v_t_j, self.all_t.unsqueeze(1), self.all_i.unsqueeze(0)).amax(2).clamp(min=self.u_0)
        if return_mu_t_i_j:
            mu_t_i_j = mu_t_i[:,:,None] == self.all_j[None,None,:]
            return u_t_i, v_t_j, mu_t_i_j

        return u_t_i, v_t_j, mu_t_i


#    # Reverse auction methods
#     def _reverse_bid(self, j_id, u_i, eps):
#         # Compute top 2 values and indices at current prices

#         V_i_j = self.get_V_i_j(u_i, j_id)
#         top2 = V_i_j.topk(2, dim=0)

#         # Filter out items preferring the outside option
#         bidding = top2.values[0] >= self.v_0 + eps
#         bidder_id, out_id = j_id[bidding], j_id[~bidding]

#         # Compute selected agent and second best value for each item
#         i_j = top2.indices[0, bidding]
#         w_j = top2.values[1, bidding]

#         # Compute bids
#         # bid_j = self.get_U_i_j(w_j - eps, i_j, bidder_id)
#         bid_j = self.get_U_i_j(w_j, i_j, bidder_id) + eps

#         return out_id, bidder_id, i_j, bid_j

#     def _reverse_assign(self, bidder_id, i_j, bid_j):
#         unique_id, inverse = i_j.unique(return_inverse=True)

#         best_bid = torch.empty(len(unique_id), dtype=bid_j.dtype, device=self.device)
#         best_bid.scatter_reduce_(0, inverse, bid_j, reduce='amax', include_self=False)

#         is_best = (bid_j == best_bid[inverse])
#         winner = torch.empty(len(unique_id), dtype=bidder_id.dtype, device=self.device)
#         winner[inverse[is_best]] = bidder_id[is_best]

#         return unique_id, winner, best_bid

#     def _reverse_iteration(self, unmatched_j, u_i, mu_j, eps):
#         out_id, bidder_id, i_j, bid_j = self._reverse_bid(unmatched_j, u_i, eps)
#         mu_j[out_id] = self.num_i

#         unique_id, winner, best_bid = self._reverse_assign(bidder_id, i_j, bid_j)

#         reset_j = torch.isin(mu_j, unique_id)
#         mu_j[reset_j] = -1
#         mu_j[winner] = unique_id

#         u_i[unique_id] = best_bid
#         unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]#.contiguous()

#         return unmatched_j, u_i, mu_j

#     def reverse_auction(self, init_u_i=None, init_mu_j=None, eps=0, return_mu_i_j = False):
#         u_i = self.init_u_i.clone() if init_u_i is None else init_u_i
#         mu_j = self.init_mu_j.clone() if init_mu_j is None else init_mu_j
#         unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]

#         while unmatched_j.numel() > 0:
#             if self.method == "GS":
#                 unmatched_j = unmatched_j[:1]
#             elif self.method == "batched":
#                 batch_size = max(1, int(self.sampling_rate * unmatched_j.size(0)))
#                 batch = unmatched_j[torch.randperm(unmatched_j.size(0))[:batch_size]]

#             unmatched_j, u_i, mu_j = self._reverse_iteration(unmatched_j, u_i, mu_j, eps)

#         v_j = self.get_V_i_j(u_i, j_id = self.all_j).amax(dim=0).clamp(min=self.v_0)

#         if return_mu_i_j:
#             mu_i_j = mu_j.unsqueeze(0) == self.all_i.unsqueeze(1)
#             return u_i, v_j, mu_i_j

#         return u_i, v_j, mu_j


#     # Scaling method
#     def forward_reverse_scaling(self, eps_init, eps_target, scaling_factor):
#         eps = eps_init
#         v_j = self.init_v_j.clone()

#         while True:
#             u_i, v_j, mu_i  = self.forward_auction(init_v_j = v_j,  eps= eps)
#             eps *=  scaling_factor
#             u_i, v_j, mu_j  = self.reverse_auction(init_u_i = u_i, eps= eps)
#             if eps <= eps_target:
#                 break

#         u_i, v_j, mu_i_j  = self.reverse_auction(init_u_i = u_i, eps= eps, return_mu_i_j= True)
#         mu_i = torch.where(mu_i_j.any(dim=1), mu_i_j.int().argmax(dim=1), -1)
#         violations_i = (u_i > self.u_0) & (mu_i_j.sum(dim=1) == 0)
#         mu_i[violations_i] = -1

#         u_i, v_j, mu_i_j  = self.forward_auction(init_v_j = v_j, init_mu_i= mu_i, eps= eps, return_mu_i_j= True)

#         return u_i, v_j, mu_i_j








