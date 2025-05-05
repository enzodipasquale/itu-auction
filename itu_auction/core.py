import torch

class ITUauction:
    def __init__(self, num_i, num_j, get_U_i_j, get_V_i_j, lb = (0, 0)):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_i = num_i
        self.num_j = num_j

        self.all_i = torch.arange(num_i, device=self.device)
        self.all_j = torch.arange(num_j, device=self.device)

        self._get_U_i_j = get_U_i_j
        self._get_V_i_j = get_V_i_j

        self.lb_i = lb[0]
        self.lb_j = lb[1]

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
        CS = (self.get_U_i_j(v_j).amax(dim=1) - u_i).max().item()
        feas = torch.all((mu_i_j.sum(dim=1) <= 1)) and torch.all((mu_i_j.sum(dim=0) <= 1)) 
        IR_i =  torch.all(u_i[mu_i_j.sum(dim=1) == 0] <= self.lb_i + eps)
        IR_j =  torch.all(v_j[mu_i_j.sum(dim=0) == 0] <= self.lb_j + eps)

        print("=== Equilibrium Conditions ===")
        print(f"Complementary Slackness      : {CS:.4f}")
        print(f"Feasibility                  : {feas}")
        print(f"Individual Rationality (i)   : {IR_i}")
        print(f"Individual Rationality (j)   : {IR_j}")

        if IR_i == False:
            print(u_i[mu_i_j.sum(dim=1) == 0])
        if IR_j == False:
            print(v_j[mu_i_j.sum(dim=0) == 0])

        return CS, feas, IR_i, IR_j


    # Forward auction

    def _forward_bid(self, i_id, v_j, eps):

        # Compute top 2 values and indices at current prices
        U_i_j = self.get_U_i_j(v_j, i_id)
        top2 = U_i_j.topk(2, dim=1) 

        # Filter out bidders preferring the outside option
        bidding = top2.values[:, 0] > self.lb_i + eps
        bidder_id, out_id = i_id[bidding], i_id[~bidding]

        # Compute selected item and second best value for each bidder
        j_i = top2.indices[bidding, 0]
        w_i = top2.values[bidding, 1]  

        # Compute bids
        bid_i = self.get_V_i_j(w_i - eps, j_i, bidder_id)

        return out_id, bidder_id, j_i, bid_i

    # @torch.jit.trace
    def _forward_assign(self, bidder_id, j_i, bid_i):  
        unique_id, inverse = j_i.unique(return_inverse=True)

        best_bid = torch.empty(len(unique_id), dtype=bid_i.dtype, device=self.device)
        best_bid.scatter_reduce_(0, inverse, bid_i, reduce='amax', include_self=False)

        is_best = bid_i == best_bid[inverse]
        winner = torch.empty(len(unique_id),  dtype=bidder_id.dtype, device=self.device)
        winner[inverse[is_best]] = bidder_id[is_best]

        return unique_id, winner, best_bid

    def _forward_iteration(self, unmatched_i, v_j, mu_i, eps):
        # Bidding phase
        out_id, bidder_id, j_i, bid_i = self._forward_bid(unmatched_i, v_j, eps)
        mu_i[out_id] = self.num_j

        # Assignment phase
        unique_id, winner, best_bid = self._forward_assign(bidder_id, j_i, bid_i)

        # Update assignment
        mask = (mu_i.unsqueeze(1) == unique_id.unsqueeze(0)).any(dim=1)
        mu_i[mask] = -1 
        mu_i[winner] = unique_id

        # mu_i[mu_j[unique_id]] = -1
        # mu_i[winner] = unique_id
        # mu_j[unique_id] = winner

        # Update prices
        v_j[unique_id] = best_bid

        # Update unmatched bidders
        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0].contiguous()

        return unmatched_i, v_j, mu_i

    def forward_auction(self, init_v_j= None, init_mu_i= None, eps = 0):

        # Initialize prices and partial assignment
        v_j = torch.full((self.num_j,), self.lb_j, dtype=torch.float, device=self.device) if init_v_j is None else init_v_j
        mu_i = torch.full((self.num_i,), -1, dtype=torch.long, device=self.device) if init_mu_i is None else init_mu_i
        # mu_j = torch.full((self.num_j,), -1, dtype=torch.long, device=self.device) if init_mu_j is None else init_mu_j
        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0].contiguous()

        # Iterate until all bidders are matched
        while unmatched_i.numel() > 0:
            unmatched_i, v_j, mu_i = self._forward_iteration(unmatched_i, v_j, mu_i, eps)

        # Compute utility for each bidder and binary assignment matrix
        u_i = self.get_U_i_j(v_j).amax(dim=1).clamp(min=self.lb_i)
        mu_i_j = mu_i.unsqueeze(1) == self.all_j.unsqueeze(0)

        return u_i, v_j, mu_i_j



   # Reverse auction methods
    def _reverse_bid(self, j_id, u_i, eps):
        # Compute top 2 values and indices at current prices

        V_i_j = self.get_V_i_j(u_i, j_id)
        top2 = V_i_j.topk(2, dim=0)

        # Filter out items preferring the outside option
        bidding = top2.values[0] > self.lb_j + eps
        bidder_id, out_id = j_id[bidding], j_id[~bidding]

        # Compute selected agent and second best value for each item
        i_j = top2.indices[0, bidding]
        w_j = top2.values[1, bidding]

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

        mask  = (mu_j.unsqueeze(1) == unique_id.unsqueeze(0)).any(dim=1)
        mu_j[mask] = -1
        mu_j[winner] = unique_id

        u_i[unique_id] = best_bid
        unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0].contiguous()

        return unmatched_j, u_i, mu_j


    def reverse_auction(self, init_u_i=None, init_mu_j=None, eps=0):
        u_i = torch.full((self.num_i,), self.lb_i, dtype=torch.float, device=self.device) if init_u_i is None else init_u_i
        mu_j = torch.full((self.num_j,), -1, dtype=torch.long, device=self.device) if init_mu_j is None else init_mu_j
        unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0].contiguous()

        while unmatched_j.numel() > 0:
            unmatched_j, u_i, mu_j = self._reverse_iteration(unmatched_j, u_i, mu_j, eps)

        v_j = self.get_V_i_j(u_i).amax(dim=0).clamp(min=self.lb_j)
        mu_i_j = mu_j.unsqueeze(0) == self.all_i.unsqueeze(1)

        return u_i, v_j, mu_i_j


    # Scaling method

    # def forward_auction_scaling(self, eps_init, eps_target, scaling_factor):
    #     eps = eps_init
    #     v_j = torch.full((self.num_j,), self.lb_j, dtype=torch.float, device=self.device)
        
    #     while True:
    #         u_i, v_j, mu_i_j = self.forward_auction(init_v_j = v_j, eps = eps)
    #         if eps <= eps_target:
    #             break
    #         eps *= scaling_factor

    #     return u_i, v_j, mu_i_j


    def forward_reverse_scaling(self, eps_init, eps_target, scaling_factor):
        eps = eps_init
        v_j = torch.full((self.num_j,), self.lb_j, dtype=torch.float, device=self.device)
        mu_i = torch.full((self.num_i,), -1, dtype=torch.long, device=self.device) 
        
        while True:
            u_i, v_j, mu_i_j = self.forward_auction(init_v_j= v_j, eps = eps)
            if eps <= eps_target:
                break
            eps *= scaling_factor
            u_i, v_j, mu_i_j = self.reverse_auction(init_u_i= u_i, eps = eps)

        return u_i, v_j, mu_i_j








