import torch

class ITUauction:
    def __init__(self, num_i, num_j, get_U_i_j, get_V_i_j, lb = (0, 0)):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_i = num_i
        self.num_j = num_j

        self.all_i = torch.arange(num_i)
        self.all_j = torch.arange(num_j)

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

    def get_V_i_j(self, u_i, j_id, i_id = None):
        j_id = j_id if j_id is not None else self.all_j

        if i_id is None:
            return self._get_V_i_j(u_i, j_id)
        else:
            return self._get_V_i_j(u_i, j_id, i_id)

        

    def forward_bid(self, i_id, v_j, tol_ε):

        # Compute top 2 values and indices at current prices
        U_i_j = self.get_U_i_j(v_j, i_id)
        top2 = U_i_j.topk(2, dim=1) 

        # Filter out bidders preferring the outside option
        bidding = top2.values[:, 0] > self.lb_i + tol_ε
        bidder_id, out_id = i_id[bidding], i_id[~bidding]

        # Compute selected item and second best value for each bidder
        j_i = top2.indices[bidding, 0]
        w_i = top2.values[bidding, 1]  

        # Compute bids
        bid_i = self.get_V_i_j(w_i - tol_ε, j_i, i_id)

        return out_id, bidder_id, j_i, bid_i

    # def forward_assign(bidder_id, j_i, bid_i):  





    def forward_auction(self, init_v_j= None, init_mu_i= None, tol_ε = 0):

        v_j = init_v_j or torch.full((self.num_j,), self.lb_j, dtype=torch.float, device=self.device)
        mu_i = init_mu_i if init_mu_i is not None else torch.full((self.num_i,), -1, dtype=torch.long, device=self.device)

        unmatched_i = torch.where(mu_i == -1)[0]

        while len(unmatched_i) > 0:
            # Bidding phase
            out_id, bidder_id, j_i, bid_i = self.forward_bid(unmatched_i, v_j)
            mu_i[out_id] = self.num_j

            # Assignment phase














              


        return 0










