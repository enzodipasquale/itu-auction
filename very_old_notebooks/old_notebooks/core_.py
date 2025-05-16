import torch

class ITUauction:
    def __init__(self, num_i, num_j, get_U_i_j, get_V_i_j, v_0 = 0):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_i = num_i
        self.num_j = num_j

        self.all_i = torch.arange(num_i, device=self.device)
        self.all_j = torch.arange(num_j, device=self.device)

        self._get_U_i_j = get_U_i_j
        self._get_V_i_j = get_V_i_j

        self.v_0 = v_0
        self.v_j_init = torch.full((self.num_j,), self.v_0, dtype=torch.float, device=self.device)
        self.u_i_init = torch.full((self.num_i,), - torch.inf , dtype=torch.float, device=self.device)
        self.mu_i_init = torch.full((self.num_i,), -1, dtype=torch.long, device=self.device)
        self.mu_j_init = torch.full((self.num_j,), -1, dtype=torch.long, device=self.device)

    def get_U_i_j(self, v_j, i_id = None, j_id = None):
        i_id = i_id if i_id is not None else self.all_i

        if j_id is None:
            return self._get_U_i_j(v_j, i_id)
        else:
            return self._get_U_i_j(v_j, i_id, j_id)

    def get_V_i_j(self, u_i, j_id = None, i_id = None):
        j_id = j_id if j_id is not None else self.all_j

        if i_id is None:
            return self._get_V_i_j(u_i, j_id)
        else:
            return self._get_V_i_j(u_i, j_id, i_id)


    def check_equilibrium(self, u_i, v_j, mu_i_j, eps = 0):
        CS = (self.get_U_i_j(v_j).amax(dim=1) - u_i).amax()
        feas = torch.all((mu_i_j.sum(dim=1) == 1)) and torch.all((mu_i_j.sum(dim=0) <= 1)) 
        IR_j =  torch.all(v_j[mu_i_j.sum(dim=0) == 0] <= self.v_0 + eps)

        print("=== Equilibrium Conditions ===")
        print(f"Complementary Slackness      : {CS:.4f}")
        print(f"Feasibility                  : {feas}")
        print(f"Individual Rationality (j)   : {IR_j}")

        if IR_j == False:
            print(v_j[mu_i_j.sum(dim=0) == 0] )

        return CS, feas, IR_j


    # Forward auction

    def _forward_bid(self, i_id, v_j, eps):

        # Compute top 2 values and indices at current prices
        U_i_j = self.get_U_i_j(v_j, i_id)
        top2 = U_i_j.topk(2, dim=1) 

        # Compute selected item and second best value for each bidder
        j_i = top2.indices[:, 0]
        w_i = top2.values[:, 1]  

        # Compute bids
        bid_i = self.get_V_i_j(w_i - eps, j_i, i_id)

        return i_id, j_i, bid_i, w_i

    def _forward_assign(self, i_id, j_i, bid_i):  
        j_demanded, inverse = j_i.unique(return_inverse=True)

        best_bid = torch.empty(len(j_demanded), dtype=bid_i.dtype, device=self.device)
        best_bid.scatter_reduce_(0, inverse, bid_i, reduce='amax', include_self=False)

        is_best = bid_i == best_bid[inverse]
        winner = torch.empty(len(j_demanded),  dtype=i_id.dtype, device=self.device)
        winner[inverse[is_best]] = i_id[is_best]

        return j_demanded, winner, best_bid

    def _forward_iteration(self, unmatched_i, u_i ,v_j, mu_i, mu_j, eps):
        # Bidding phase
        i_id, j_i, bid_i, w_i = self._forward_bid(unmatched_i, v_j, eps)

        # Assignment phase
        j_demanded, winner, best_bid = self._forward_assign(i_id, j_i, bid_i)

        # Update values
        v_j[j_demanded] = best_bid.clamp(min=self.v_0)
        u_i[winner] = self.get_U_i_j(best_bid, winner, j_demanded)

        # Update assignment
        valid_bid = best_bid >= self.v_0
        j_demanded = j_demanded[valid_bid]
        winner = winner[valid_bid]

        current_match = mu_j[j_demanded]
        current_match = current_match[current_match != -1]
        mu_i[current_match] = -1
        mu_i[winner] = j_demanded
        mu_j[j_demanded] = winner

        return u_i, v_j, mu_i, mu_j

    def forward_auction(self, init_u_i = None, init_v_j= None, init_mu_i= None, init_mu_j= None, eps = 0):

        if self.num_j < self.num_i:
            raise ValueError("Number of items must be greater than number of bidders")

        # Initialize prices and partial assignment
        v_j = self.v_j_init.clone() if init_v_j is None else init_v_j
        u_i = self.u_i_init.clone() if init_u_i is None else init_u_i
        mu_i = self.mu_i_init.clone() if init_mu_i is None else init_mu_i
        mu_j = self.mu_j_init.clone() if init_mu_j is None else init_mu_j

        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0].contiguous()

        # Iterate until all bidders are matched
        while unmatched_i.numel() > 0:
            # unmatched_i = unmatched_i[:1]
            u_i, v_j, mu_i, mu_j = self._forward_iteration(unmatched_i, u_i, v_j, mu_i, mu_j, eps)
            unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0].contiguous()

        # Compute utility for each bidder and binary assignment matrix
        # u_i = self.get_U_i_j(v_j).amax(dim=1)
        mu_i_j = mu_i.unsqueeze(1) == self.all_j.unsqueeze(0)

        return u_i, v_j, mu_i_j


    

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
#         bid_j = self.get_U_i_j(w_j - eps, i_j, bidder_id)

#         return out_id, bidder_id, i_j, bid_j, w_j

#     def _reverse_assign(self, bidder_id, i_j, bid_j):
#         i_demanded, inverse = i_j.unique(return_inverse=True)

#         best_bid = torch.empty(len(i_demanded), dtype=bid_j.dtype, device=self.device)
#         best_bid.scatter_reduce_(0, inverse, bid_j, reduce='amax', include_self=False)

#         is_best = bid_j == best_bid[inverse]
#         winner = torch.empty(len(i_demanded), dtype=bidder_id.dtype, device=self.device)
#         winner[inverse[is_best]] = bidder_id[is_best]

#         return i_demanded, winner, best_bid, is_best


#     def _reverse_iteration(self, unmatched_j, u_i ,v_j, mu_i, mu_j, eps):

#         out_id, bidder_id, i_j, bid_j, w_j = self._reverse_bid(unmatched_j, u_i, eps)

#         i_demanded, winner, best_bid, is_best = self._reverse_assign(bidder_id, i_j, bid_j)

#         # Update values
#         v_j[winner] = (best_bid - eps).clamp(min=self.v_0)
#         u_i[i_demanded] = self.get_U_i_j(v_j[winner], i_demanded, winner)

#         # Update assignment
#         current_match = mu_i[i_demanded]
#         current_match = current_match[current_match != -1]
#         mu_j[current_match] = -1
#         mu_j[winner] = i_demanded
#         mu_i[i_demanded] = winner

#         return u_i, v_j, mu_i, mu_j







    def _reverse_iteration(self, unmatched_j, u_i ,v_j, mu_i, mu_j, eps):
        # Pick j
        j = next((j for j in unmatched_j if v_j[j] > self.v_0), None)
        # Compute top 2 values and indices at current prices
        V_i_j = self.get_V_i_j(u_i, [j])
        top2 = V_i_j.topk(2, dim=0)

        beta_j = top2.values[0]
        w_j = top2.values[1]
        i_j = top2.indices[0]

        if beta_j >= self.v_0 + eps:
            v_j[j] = (w_j - eps).clamp(min=self.v_0)
            u_i[i_j] = self.get_U_i_j(v_j[j], i_j, [j])

            if mu_i[i_j] != -1:
                mu_j[mu_i[i_j]] = -1
            mu_i[i_j] = j
            mu_j[j] = i_j
        else:
            v_j[j] = beta_j - eps




    def _forward_auction_cycle(self, u_i ,v_j, mu_i, mu_j, eps):
        print("Forward auction cycle")
        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0].contiguous()
        init_num_unmatched_i = unmatched_i.numel()
        while True:
            u_i, v_j, mu_i, mu_j = self._forward_iteration(unmatched_i, u_i, v_j, mu_i, mu_j, eps)
            unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0]
            if unmatched_i.numel() < init_num_unmatched_i:
                break
        
        if unmatched_i.numel() > 0:
            self._reverse_auction_cycle(u_i ,v_j, mu_i, mu_j, eps)
        else:
            self._reverse_auction(u_i ,v_j, mu_i, mu_j, eps)

    def _reverse_auction_cycle(self, u_i ,v_j, mu_i, mu_j, eps):
        unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0].contiguous()
        init_num_unmatched_j = unmatched_j.numel()
        stop = unmatched_j.numel() < init_num_unmatched_j or torch.all(v_j[unmatched_j] <= self.v_0 )
        print("Reverse auction cycle")
        while not stop:
            u_i, v_j, mu_i, mu_j = self._reverse_iteration(unmatched_j, u_i, v_j, mu_i, mu_j, eps)
            unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]
            stop = unmatched_j.numel() < init_num_unmatched_j or torch.all(v_j[unmatched_j] <= self.v_0 )
           
        
        print('done')
        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0]
        if unmatched_i.numel() > 0:
            self._forward_auction_cycle(u_i ,v_j, mu_i, mu_j, eps)
        else:
            self._reverse_auction(u_i ,v_j, mu_i, mu_j, eps)


    def _reverse_auction(self, u_i ,v_j, mu_i, mu_j, eps):
        print("Reverse auction")
        unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0].contiguous()
        init_num_unmatched_j = unmatched_j.numel()

        while v_j[unmatched_j].max() > self.v_0:
            u_i, v_j, mu_i, mu_j = self._reverse_iteration(unmatched_j, u_i, v_j, mu_i, mu_j, eps)
            unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]
        print('done')
    


    def forward_reverse_auction(self, eps):
        if self.num_j < self.num_i:
            raise ValueError("Number of items must be greater than number of bidders")

        # Initialize prices and partial assignment
        v_j = self.v_j_init.clone() 
        u_i = self.u_i_init.clone() 
        mu_i = self.mu_i_init.clone() 
        mu_j = self.mu_j_init.clone() 

        self._forward_auction_cycle(u_i ,v_j, mu_i, mu_j, eps)

        mu_i_j = mu_i.unsqueeze(1) == self.all_j.unsqueeze(0)

        # u_i, v_j, mu_i_j = self.forward_auction(init_v_j= v_j, eps = eps)
        return u_i ,v_j, mu_i_j

        


























