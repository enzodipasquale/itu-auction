import torch
import sys

class ITUauction:
    """
    Batched Imperfectly Transferable Utility (ITU) Auction Algorithm Implementation.
    
    This class implements batched auction algorithms for solving multiple assignment problems
    simultaneously. It extends the single-problem ITUauction class to handle num_t parallel
    auction problems, providing significant speedup through GPU parallelization.
    
    The batched version processes multiple auction problems in parallel, where each problem
    has the same number of bidders and items but different valuation functions. This is
    particularly useful for:
    - Monte Carlo simulations
    - Sensitivity analysis
    - Parameter estimation
    - Large-scale market simulations
    
    Attributes:
        num_i (int): Number of bidders/agents per problem
        num_j (int): Number of items per problem  
        num_t (int): Number of parallel auction problems
        device (torch.device): Computation device (CPU or CUDA)
        u_0 (float): Lower bound for bidder utilities (outside option)
        v_0 (float): Lower bound for item values (outside option)
        method (str): Auction method ('GS' for Gauss-Seidel, 'batched' for batched processing)
        sampling_rate (float): Sampling rate for batched processing (0-1)
    
    Example:
        >>> # Define batched valuation functions
        >>> def get_U_t_i_j(v_t_j, t_id, i_id):
        ...     return Φ_t_i_j[t_id, i_id] - v_t_j[t_id]  # Batched bidder utilities
        >>> def get_V_t_i_j(u_t_i, t_id, j_id):
        ...     return Φ_t_i_j[t_id, :, j_id] - u_t_i[t_id]  # Batched item values
        >>> 
        >>> # Create batched auction instance
        >>> auction = ITUauction(num_i=100, num_j=98, num_t=50, 
        ...                      get_U_t_i_j=get_U_t_i_j, get_V_t_i_j=get_V_t_i_j)
        >>> 
        >>> # Solve all problems simultaneously
        >>> u_t_i, v_t_j, mu_t_i_j = auction.forward_reverse_scaling(
        ...     eps_init=50, eps_target=1e-4, scaling_factor=0.5
        ... )
    """
    
    def __init__(self, num_i, num_j, num_t, get_U_t_i_j, get_V_t_i_j, lb = (0, 0)):
        """
        Initialize the batched ITU auction instance.
        
        Args:
            num_i (int): Number of bidders/agents per problem
            num_j (int): Number of items per problem
            num_t (int): Number of parallel auction problems
            get_U_t_i_j (callable): Function computing batched bidder utilities
                                   Signature: get_U_t_i_j(v_t_j, t_id, i_id) -> torch.Tensor
                                   Returns: Utility tensor of shape (len(t_id), len(i_id), num_j)
            get_V_t_i_j (callable): Function computing batched item values
                                   Signature: get_V_t_i_j(u_t_i, t_id, j_id) -> torch.Tensor
                                   Returns: Value tensor of shape (len(t_id), num_i, len(j_id))
            lb (tuple, optional): Lower bounds for utilities and values (u_0, v_0). 
                                Defaults to (0, 0).
        
        Note:
            The batched valuation functions should handle the additional time dimension t.
            All tensors have an extra dimension for the batch/time index.
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
        """
        Check if the current batched allocation satisfies market equilibrium conditions.
        
        This method verifies equilibrium conditions across all parallel auction problems.
        The conditions are checked for each problem t independently.
        
        Args:
            u_t_i (torch.Tensor): Bidder utilities of shape (num_t, num_i)
            v_t_j (torch.Tensor): Item values of shape (num_t, num_j)
            mu_t_i_j (torch.Tensor): Binary assignment tensor of shape (num_t, num_i, num_j)
            eps (float, optional): Tolerance for equilibrium conditions. Defaults to 0.
        
        Returns:
            tuple: (CS, feas, IR_i, IR_j) where:
                - CS (float): Maximum complementary slackness violation across all problems
                - feas (bool): Whether feasibility constraints are satisfied for all problems
                - IR_i (bool): Whether bidder individual rationality holds for all problems
                - IR_j (bool): Whether item individual rationality holds for all problems
        
        Example:
            >>> u_t_i, v_t_j, mu_t_i_j = auction.forward_auction(return_mu_t_i_j=True)
            >>> CS, feas, IR_i, IR_j = auction.check_equilibrium(u_t_i, v_t_j, mu_t_i_j, eps=1e-4)
            >>> print(f"All problems in equilibrium: {CS <= 1e-4 and feas and IR_i and IR_j}")
        """
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
        """
        Compute bids in the batched forward auction phase.
        
        Each unmatched bidder across all problems computes their optimal bid based on
        current item prices. The bidding is processed in parallel across all problems.
        
        Args:
            unmatched_t_i (tuple): (t_id, i_id) indices of unmatched bidders across all problems
            v_t_j (torch.Tensor): Current item prices of shape (num_t, num_j)
            eps (float): Epsilon parameter for bid computation
        
        Returns:
            tuple: (out_t_i, bidder_t_i_id, j_ti, bid_ti) where:
                - out_t_i: Bidders who prefer outside option
                - bidder_t_i_id: Bidders who place bids
                - j_ti: Items chosen by each bidder
                - bid_ti: Bid amounts for each item
        """
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
        """
        Assign items to bidders based on their bids in batched auction.
        
        For each item across all problems, the highest bidder wins. Ties are resolved
        arbitrarily. The assignment is processed in parallel.
        
        Args:
            bidder_t_i_id (tuple): (bidder_t_id, bidder_i_id) bidders who placed bids
            j_ti (torch.Tensor): Items chosen by each bidder
            bid_ti (torch.Tensor): Bid amounts for each item
        
        Returns:
            tuple: (t_id, j_id, winner_i_id, best_bid) where:
                - t_id: Problem indices for unique items
                - j_id: Unique items that received bids
                - winner_i_id: Winning bidder for each item
                - best_bid: Winning bid amount for each item
        """
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
        """
        Perform one iteration of the batched forward auction.
        
        This includes the bidding phase (where bidders compute optimal bids)
        and the assignment phase (where items are assigned to highest bidders).
        All operations are performed in parallel across all problems.
        
        Args:
            unmatched_t_i (tuple): Currently unmatched bidders across all problems
            v_t_j (torch.Tensor): Current item prices of shape (num_t, num_j)
            mu_t_i (torch.Tensor): Current bidder assignments of shape (num_t, num_i)
            mu_t_j (torch.Tensor): Current item assignments of shape (num_t, num_j)
            eps (float): Epsilon parameter
        
        Returns:
            tuple: (unmatched_t_i, v_t_j, mu_t_i, mu_t_j) updated after one iteration
        """
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
        """
        Run the batched forward auction algorithm to find equilibrium allocations.
        
        In the batched forward auction, bidders compete for items across all parallel
        problems simultaneously. The algorithm iteratively updates item prices and
        assignments until all bidders are matched or prefer their outside option.
        
        Args:
            init_v_t_j (torch.Tensor, optional): Initial item prices of shape (num_t, num_j).
                                               If None, uses default prices. Defaults to None.
            init_mu_t_i (torch.Tensor, optional): Initial bidder assignments of shape (num_t, num_i).
                                                If None, all bidders start unmatched. Defaults to None.
            init_mu_t_j (torch.Tensor, optional): Initial item assignments of shape (num_t, num_j).
                                                If None, all items start unmatched. Defaults to None.
            eps (float, optional): Epsilon parameter for bid computation. Defaults to 0.
            return_mu_t_i_j (bool, optional): Whether to return binary assignment tensor.
                                            Defaults to False.
        
        Returns:
            tuple: (u_t_i, v_t_j, mu_t_i, mu_t_j) or (u_t_i, v_t_j, mu_t_i_j) where:
                - u_t_i (torch.Tensor): Final bidder utilities of shape (num_t, num_i)
                - v_t_j (torch.Tensor): Final item prices of shape (num_t, num_j)
                - mu_t_i (torch.Tensor): Final bidder assignments of shape (num_t, num_i)
                - mu_t_j (torch.Tensor): Final item assignments of shape (num_t, num_j)
                - mu_t_i_j (torch.Tensor): Binary assignment tensor of shape (num_t, num_i, num_j)
                                          (only if return_mu_t_i_j=True)
        
        Example:
            >>> # Run batched forward auction
            >>> u_t_i, v_t_j, mu_t_i, mu_t_j = auction.forward_auction(eps=1e-4)
            >>> 
            >>> # Get binary assignment tensor
            >>> u_t_i, v_t_j, mu_t_i_j = auction.forward_auction(eps=1e-4, return_mu_t_i_j=True)
        """
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
        """
        Compute bids in the batched reverse auction phase.
        
        Each unmatched item across all problems computes its optimal bid based on
        current bidder utilities. The bidding is processed in parallel across all problems.
        
        Args:
            unmatched_t_j (tuple): (t_id, j_id) indices of unmatched items across all problems
            u_t_i (torch.Tensor): Current bidder utilities of shape (num_t, num_i)
            eps (float): Epsilon parameter for bid computation
        
        Returns:
            tuple: (out_t_j, bidder_t_j_id, i_tj, bid_tj) where:
                - out_t_j: Items that prefer outside option
                - bidder_t_j_id: Items that place bids
                - i_tj: Bidders chosen by each item
                - bid_tj: Bid amounts for each bidder
        """
        t_id, j_id = unmatched_t_j
        
        # Compute top 2 values and indices at current prices
        V_t_i_j = self.get_V_t_i_j(u_t_i, t_id, j_id)
        top2 = V_t_i_j.topk(2, dim=-1) 

        # Filter out items preferring the outside option
        bidding = top2.values[:, 0] >= self.v_0 + eps
        bidder_t_j_id = bidder_t_id, bidder_j_id = t_id[bidding], j_id[bidding]
        out_t_i = t_id[~bidding], j_id[~bidding]
        
        # Compute selected agent and second best value for each item
        i_tj = top2.indices[bidding, 0]
        w_tj = torch.clamp(top2.values[bidding, 1], min=self.v_0)

        # Compute bids
        bid_tj = self.get_U_t_i_j(w_tj , bidder_t_id, i_tj, bidder_j_id) + eps
        return out_t_i, bidder_t_j_id, i_tj, bid_tj

    def _reverse_assign(self, bidder_t_j_id, i_tj, bid_tj):
        """
        Assign bidders to items based on their bids in batched reverse auction.
        
        For each bidder across all problems, the highest bidding item wins. Ties are
        resolved arbitrarily. The assignment is processed in parallel.
        
        Args:
            bidder_t_j_id (tuple): (bidder_t_id, bidder_j_id) items that placed bids
            i_tj (torch.Tensor): Bidders chosen by each item
            bid_tj (torch.Tensor): Bid amounts for each bidder
        
        Returns:
            tuple: (t_id, i_id, winner_j_id, best_bid) where:
                - t_id: Problem indices for unique bidders
                - i_id: Unique bidders that received bids
                - winner_j_id: Winning item for each bidder
                - best_bid: Winning bid amount for each bidder
        """
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
        """
        Perform one iteration of the batched reverse auction.
        
        This includes the bidding phase (where items compute optimal bids)
        and the assignment phase (where bidders are assigned to highest bidding items).
        All operations are performed in parallel across all problems.
        
        Args:
            unmatched_t_j (tuple): Currently unmatched items across all problems
            u_t_i (torch.Tensor): Current bidder utilities of shape (num_t, num_i)
            mu_t_i (torch.Tensor): Current bidder assignments of shape (num_t, num_i)
            mu_t_j (torch.Tensor): Current item assignments of shape (num_t, num_j)
            eps (float): Epsilon parameter
        
        Returns:
            tuple: (unmatched_t_j, u_t_i, mu_t_i, mu_t_j) updated after one iteration
        """
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
        """
        Run the batched reverse auction algorithm to find equilibrium allocations.
        
        In the batched reverse auction, items compete for bidders across all parallel
        problems simultaneously. The algorithm iteratively updates bidder utilities and
        assignments until all items are matched or prefer their outside option.
        
        Args:
            init_u_t_i (torch.Tensor, optional): Initial bidder utilities of shape (num_t, num_i).
                                               If None, uses default utilities. Defaults to None.
            init_mu_t_i (torch.Tensor, optional): Initial bidder assignments of shape (num_t, num_i).
                                                If None, all bidders start unmatched. Defaults to None.
            init_mu_t_j (torch.Tensor, optional): Initial item assignments of shape (num_t, num_j).
                                                If None, all items start unmatched. Defaults to None.
            eps (float, optional): Epsilon parameter for bid computation. Defaults to 0.
            return_mu_t_i_j (bool, optional): Whether to return binary assignment tensor.
                                            Defaults to False.
        
        Returns:
            tuple: (u_t_i, v_t_j, mu_t_j) or (u_t_i, v_t_j, mu_t_i_j) where:
                - u_t_i (torch.Tensor): Final bidder utilities of shape (num_t, num_i)
                - v_t_j (torch.Tensor): Final item prices of shape (num_t, num_j)
                - mu_t_j (torch.Tensor): Final item assignments of shape (num_t, num_j)
                - mu_t_i_j (torch.Tensor): Binary assignment tensor of shape (num_t, num_i, num_j)
                                          (only if return_mu_t_i_j=True)
        
        Example:
            >>> # Run batched reverse auction
            >>> u_t_i, v_t_j, mu_t_j = auction.reverse_auction(eps=1e-4)
            >>> 
            >>> # Get binary assignment tensor
            >>> u_t_i, v_t_j, mu_t_i_j = auction.reverse_auction(eps=1e-4, return_mu_t_i_j=True)
        """
        u_t_i = self.init_u_t_i.clone() if init_u_t_i is None else init_u_t_i
        mu_t_i = self.init_mu_t_i.clone() if init_mu_t_i is None else init_mu_t_i
        mu_t_j = self.init_mu_t_j.clone() if init_mu_t_j is None else init_mu_t_j
        unmatched_t_j = (mu_t_j == -1).nonzero(as_tuple=True)

        while unmatched_t_j[0].numel() > 0:
            if self.method == "GS":
                unmatched_t_j = unmatched_t_j[:1]
            elif self.method == "batched":
                batch_size = max(1, int(self.sampling_rate * unmatched_t_j[0].size(0)))
                batch = unmatched_t_j[0][torch.randperm(unmatched_t_j[0].size(0))[:batch_size]]

            unmatched_t_j, u_t_i, mu_t_i, mu_t_j = self._reverse_iteration(unmatched_t_j, u_t_i, mu_t_i, mu_t_j, eps)

        v_t_j = self.get_V_t_i_j(u_t_i, j_id = self.all_j).amax(dim=0).clamp(min=self.v_0)

        if return_mu_t_i_j:
            mu_t_i_j = mu_t_j.unsqueeze(0) == self.all_i.unsqueeze(1)
            return u_t_i, v_t_j, mu_t_i_j

        return u_t_i, v_t_j, mu_t_j
