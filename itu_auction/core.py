import torch

class ITUauction:
    """
    Imperfectly Transferable Utility (ITU) Auction Algorithm Implementation.
    
    This class implements auction algorithms for solving assignment problems with non-separable
    valuations, where agents may have imperfectly transferable utility. The algorithm supports
    both forward and reverse auction procedures with epsilon-scaling for improved convergence.
    
    The auction mechanism finds approximate equilibrium allocations where:
    - Complementary slackness conditions are satisfied
    - Feasibility constraints are met (at most one item per bidder, at most one bidder per item)
    - Individual rationality constraints hold
    
    Attributes:
        num_i (int): Number of bidders/agents
        num_j (int): Number of items
        device (torch.device): Computation device (CPU or CUDA)
        u_0 (float): Lower bound for bidder utilities (outside option)
        v_0 (float): Lower bound for item values (outside option)
        method (str): Auction method ('GS' for Gauss-Seidel, 'batched' for batched processing)
        sampling_rate (float): Sampling rate for batched processing (0-1)
    
    Example:
        >>> # Define valuation functions
        >>> def get_U_i_j(v_j, i_id):
        ...     return Φ_i_j[i_id] - v_j.unsqueeze(0)  # Bidder utilities
        >>> def get_V_i_j(u_i, j_id):
        ...     return Φ_i_j[:, j_id] - u_i.unsqueeze(1)  # Item values
        >>> 
        >>> # Create auction instance
        >>> auction = ITUauction(num_i=100, num_j=98, get_U_i_j=get_U_i_j, get_V_i_j=get_V_i_j)
        >>> 
        >>> # Solve using epsilon-scaling
        >>> u_i, v_j, mu_i_j = auction.forward_reverse_scaling(eps_init=50, eps_target=1e-4, scaling_factor=0.5)
    """
    
    def __init__(self, num_i, num_j, get_U_i_j, get_V_i_j, lb = (0, 0)):
        """
        Initialize the ITU auction instance.
        
        Args:
            num_i (int): Number of bidders/agents
            num_j (int): Number of items
            get_U_i_j (callable): Function computing bidder utilities U(i,j) given item prices v_j
                                 Signature: get_U_i_j(v_j, i_id) -> torch.Tensor
                                 Returns: Utility matrix of shape (len(i_id), num_j)
            get_V_i_j (callable): Function computing item values V(i,j) given bidder utilities u_i
                                 Signature: get_V_i_j(u_i, j_id) -> torch.Tensor  
                                 Returns: Value matrix of shape (num_i, len(j_id))
            lb (tuple, optional): Lower bounds for utilities and values (u_0, v_0). 
                                Defaults to (0, 0).
        
        Note:
            The valuation functions should be compatible with PyTorch tensors and support
            batched operations for efficient computation. The functions should handle the
            case where i_id or j_id are None (indicating all bidders/items).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_i = num_i
        self.num_j = num_j

        self.all_i = torch.arange(num_i, device=self.device)
        self.all_j = torch.arange(num_j, device=self.device)

        self.u_0 = lb[0]
        self.v_0 = lb[1]

        self.init_v_j = torch.full((self.num_j,), self.v_0, dtype=torch.float, device=self.device)
        self.init_u_i = torch.full((self.num_i,), self.u_0 , dtype=torch.float, device=self.device)
        self.init_mu_i = torch.full((self.num_i,), -1, dtype=torch.long, device=self.device)
        self.init_mu_j = torch.full((self.num_j,), -1, dtype=torch.long, device=self.device)

        self.method = None
        self.sampling_rate = None
    
        ITUauction.get_U_i_j = staticmethod(get_U_i_j)
        ITUauction.get_V_i_j = staticmethod(get_V_i_j)


    def check_equilibrium(self, u_i, v_j, mu_i_j, eps = 0):
        """
        Check if the current allocation satisfies market equilibrium conditions.
        
        This method verifies four key equilibrium conditions:
        1. Complementary Slackness: Agents are assigned to their most preferred items
        2. Feasibility: Each agent gets at most one item, each item goes to at most one agent
        3. Individual Rationality (Bidders): Unassigned bidders get at least their outside option
        4. Individual Rationality (Items): Unassigned items get at least their outside option
        
        Args:
            u_i (torch.Tensor): Bidder utilities of shape (num_i,)
            v_j (torch.Tensor): Item values of shape (num_j,)
            mu_i_j (torch.Tensor): Binary assignment matrix of shape (num_i, num_j)
            eps (float, optional): Tolerance for equilibrium conditions. Defaults to 0.
        
        Returns:
            tuple: (CS, feas, IR_i, IR_j) where:
                - CS (float): Maximum complementary slackness violation
                - feas (bool): Whether feasibility constraints are satisfied
                - IR_i (bool): Whether bidder individual rationality holds
                - IR_j (bool): Whether item individual rationality holds
        
        Example:
            >>> u_i, v_j, mu_i_j = auction.forward_auction(return_mu_i_j=True)
            >>> CS, feas, IR_i, IR_j = auction.check_equilibrium(u_i, v_j, mu_i_j, eps=1e-4)
            >>> print(f"Equilibrium satisfied: {CS <= 1e-4 and feas and IR_i and IR_j}")
        """
        CS = (self.get_U_i_j(v_j, self.all_i).amax(dim=1) - u_i).amax()
        feas = torch.all((mu_i_j.sum(dim=1) <= 1)) and torch.all((mu_i_j.sum(dim=0) <= 1)) 
        IR_i =  torch.all(u_i[mu_i_j.sum(dim=1) == 0] <= self.u_0 + eps)
        IR_j =  torch.all(v_j[mu_i_j.sum(dim=0) == 0] <= self.v_0 + eps)

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
    def _forward_bid(self, i_id, v_j, eps):
        """
        Compute bids in the forward auction phase.
        
        Each unmatched bidder computes their optimal bid based on current item prices.
        Bidders choose their most preferred item and bid the minimum amount needed
        to win against the second-best option.
        
        Args:
            i_id (torch.Tensor): Indices of unmatched bidders
            v_j (torch.Tensor): Current item prices of shape (num_j,)
            eps (float): Epsilon parameter for bid computation
        
        Returns:
            tuple: (out_id, bidder_id, j_i, bid_i) where:
                - out_id: Bidders who prefer outside option
                - bidder_id: Bidders who place bids
                - j_i: Items chosen by each bidder
                - bid_i: Bid amounts for each item
        """
        # Compute top 2 values and indices at current prices
        U_i_j = self.get_U_i_j(v_j, i_id)
        top2 = U_i_j.topk(2, dim=1) 

        # Filter out bidders preferring the outside option
        bidding = top2.values[:, 0] >= self.u_0 + eps
        bidder_id, out_id = i_id[bidding], i_id[~bidding]

        # Compute selected item and second best value for each bidder
        j_i = top2.indices[bidding, 0]
        # w_i = top2.values[bidding, 1]  
        w_i = torch.clamp(top2.values[bidding, 1], min=self.u_0)

        # Compute bids
        bid_i = self.get_V_i_j(w_i - eps, j_i, bidder_id)

        return out_id, bidder_id, j_i, bid_i

    def _forward_assign(self, bidder_id, j_i, bid_i):  
        """
        Assign items to bidders based on their bids.
        
        For each item, the highest bidder wins. Ties are resolved arbitrarily.
        
        Args:
            bidder_id (torch.Tensor): Bidders who placed bids
            j_i (torch.Tensor): Items chosen by each bidder
            bid_i (torch.Tensor): Bid amounts for each item
        
        Returns:
            tuple: (unique_id, winner, best_bid) where:
                - unique_id: Unique items that received bids
                - winner: Winning bidder for each item
                - best_bid: Winning bid amount for each item
        """
        unique_id, inverse = j_i.unique(return_inverse=True)

        best_bid = torch.empty(len(unique_id), dtype=bid_i.dtype, device=self.device)
        best_bid.scatter_reduce_(0, inverse, bid_i, reduce='amax', include_self=False)

        is_best = (bid_i == best_bid[inverse])
        winner = torch.empty(len(unique_id),  dtype=bidder_id.dtype, device=self.device)
        winner[inverse[is_best]] = bidder_id[is_best]

        return unique_id, winner, best_bid

    def _forward_iteration(self, unmatched_i, v_j, mu_i, eps):
        """
        Perform one iteration of the forward auction.
        
        This includes the bidding phase (where bidders compute optimal bids)
        and the assignment phase (where items are assigned to highest bidders).
        
        Args:
            unmatched_i (torch.Tensor): Currently unmatched bidders
            v_j (torch.Tensor): Current item prices
            mu_i (torch.Tensor): Current bidder assignments
            eps (float): Epsilon parameter
        
        Returns:
            tuple: (unmatched_i, v_j, mu_i) updated after one iteration
        """
        # Bidding phase
        out_id, bidder_id, j_i, bid_i = self._forward_bid(unmatched_i, v_j, eps)
        mu_i[out_id] = self.num_j

        # Assignment phase
        unique_id, winner, best_bid = self._forward_assign(bidder_id, j_i, bid_i)

        # Update assignment
        reset_i = torch.isin(mu_i, unique_id)

        mu_i[reset_i] = -1
        mu_i[winner] = unique_id

        # Update prices
        v_j[unique_id] = best_bid

        # Update unmatched bidders
        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0]#.contiguous()

        return unmatched_i, v_j, mu_i

    def forward_auction(self, init_v_j= None, init_mu_i= None, eps = 0, return_mu_i_j = False):
        """
        Run the forward auction algorithm to find an equilibrium allocation.
        
        In the forward auction, bidders compete for items by placing bids. The algorithm
        iteratively updates item prices and assignments until all bidders are matched or
        prefer their outside option.
        
        Args:
            init_v_j (torch.Tensor, optional): Initial item prices. If None, uses default
                                             prices (v_0 for all items). Defaults to None.
            init_mu_i (torch.Tensor, optional): Initial bidder assignments. If None, all
                                              bidders start unmatched. Defaults to None.
            eps (float, optional): Epsilon parameter for bid computation. Defaults to 0.
            return_mu_i_j (bool, optional): Whether to return binary assignment matrix.
                                          Defaults to False.
        
        Returns:
            tuple: (u_i, v_j, mu_i) or (u_i, v_j, mu_i_j) where:
                - u_i (torch.Tensor): Final bidder utilities of shape (num_i,)
                - v_j (torch.Tensor): Final item prices of shape (num_j,)
                - mu_i (torch.Tensor): Final bidder assignments of shape (num_i,)
                - mu_i_j (torch.Tensor): Binary assignment matrix of shape (num_i, num_j)
                                        (only if return_mu_i_j=True)
        
        Example:
            >>> # Run forward auction with epsilon-scaling
            >>> u_i, v_j, mu_i = auction.forward_auction(eps=1e-4)
            >>> 
            >>> # Get binary assignment matrix
            >>> u_i, v_j, mu_i_j = auction.forward_auction(eps=1e-4, return_mu_i_j=True)
        """
        # Initialize prices and partial assignment
        v_j = self.init_v_j.clone() if init_v_j is None else init_v_j
        mu_i = self.init_mu_i.clone() if init_mu_i is None else init_mu_i
        unmatched_i = (mu_i == -1).nonzero(as_tuple=True)[0]

        # Iterate until all bidders are matched
        while unmatched_i.numel() > 0:
            if self.method == "GS":
                unmatched_i = unmatched_i[:1]
            if self.method == "batched":
                batch_size = max(1, int(self.sampling_rate * unmatched_i.size(0)))
                batch = unmatched_i[torch.randperm(unmatched_i.size(0))[:batch_size]]

            unmatched_i, v_j, mu_i = self._forward_iteration(unmatched_i, v_j, mu_i, eps)

        # Compute utility for each bidder and binary assignment matrix
        u_i = self.get_U_i_j(v_j, self.all_i).amax(dim=1).clamp(min=self.u_0)
        if return_mu_i_j:
            mu_i_j = mu_i.unsqueeze(1) == self.all_j.unsqueeze(0)
            return u_i, v_j, mu_i_j

        return u_i, v_j, mu_i


   # Reverse auction methods
    def _reverse_bid(self, j_id, u_i, eps):
        """
        Compute bids in the reverse auction phase.
        
        Each unmatched item computes its optimal bid based on current bidder utilities.
        Items choose their most preferred bidder and bid the minimum amount needed
        to win against the second-best option.
        
        Args:
            j_id (torch.Tensor): Indices of unmatched items
            u_i (torch.Tensor): Current bidder utilities of shape (num_i,)
            eps (float): Epsilon parameter for bid computation
        
        Returns:
            tuple: (out_id, bidder_id, i_j, bid_j) where:
                - out_id: Items that prefer outside option
                - bidder_id: Items that place bids
                - i_j: Bidders chosen by each item
                - bid_j: Bid amounts for each bidder
        """
        # Compute top 2 values and indices at current prices

        V_i_j = self.get_V_i_j(u_i, j_id)
        top2 = V_i_j.topk(2, dim=0)

        # Filter out items preferring the outside option
        bidding = top2.values[0] >= self.v_0 + eps
        bidder_id, out_id = j_id[bidding], j_id[~bidding]

        # Compute selected agent and second best value for each item
        i_j = top2.indices[0, bidding]
        # w_j = top2.values[1, bidding]
        w_j = torch.clamp(top2.values[1, bidding], min=self.v_0)

        # Compute bids
        # bid_j = self.get_U_i_j(w_j - eps, i_j, bidder_id)
        bid_j = self.get_U_i_j(w_j, i_j, bidder_id) + eps

        return out_id, bidder_id, i_j, bid_j

    def _reverse_assign(self, bidder_id, i_j, bid_j):
        """
        Assign bidders to items based on their bids in reverse auction.
        
        For each bidder, the highest bidding item wins. Ties are resolved arbitrarily.
        
        Args:
            bidder_id (torch.Tensor): Items that placed bids
            i_j (torch.Tensor): Bidders chosen by each item
            bid_j (torch.Tensor): Bid amounts for each bidder
        
        Returns:
            tuple: (unique_id, winner, best_bid) where:
                - unique_id: Unique bidders that received bids
                - winner: Winning item for each bidder
                - best_bid: Winning bid amount for each bidder
        """
        unique_id, inverse = i_j.unique(return_inverse=True)

        best_bid = torch.empty(len(unique_id), dtype=bid_j.dtype, device=self.device)
        best_bid.scatter_reduce_(0, inverse, bid_j, reduce='amax', include_self=False)

        is_best = (bid_j == best_bid[inverse])
        winner = torch.empty(len(unique_id), dtype=bidder_id.dtype, device=self.device)
        winner[inverse[is_best]] = bidder_id[is_best]

        return unique_id, winner, best_bid

    def _reverse_iteration(self, unmatched_j, u_i, mu_j, eps):
        """
        Perform one iteration of the reverse auction.
        
        This includes the bidding phase (where items compute optimal bids)
        and the assignment phase (where bidders are assigned to highest bidding items).
        
        Args:
            unmatched_j (torch.Tensor): Currently unmatched items
            u_i (torch.Tensor): Current bidder utilities
            mu_j (torch.Tensor): Current item assignments
            eps (float): Epsilon parameter
        
        Returns:
            tuple: (unmatched_j, u_i, mu_j) updated after one iteration
        """
        out_id, bidder_id, i_j, bid_j = self._reverse_bid(unmatched_j, u_i, eps)
        mu_j[out_id] = self.num_i

        unique_id, winner, best_bid = self._reverse_assign(bidder_id, i_j, bid_j)

        reset_j = torch.isin(mu_j, unique_id)
        mu_j[reset_j] = -1
        mu_j[winner] = unique_id

        u_i[unique_id] = best_bid
        unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]#.contiguous()

        return unmatched_j, u_i, mu_j

    def reverse_auction(self, init_u_i=None, init_mu_j=None, eps=0, return_mu_i_j = False):
        """
        Run the reverse auction algorithm to find an equilibrium allocation.
        
        In the reverse auction, items compete for bidders by placing bids. The algorithm
        iteratively updates bidder utilities and assignments until all items are matched or
        prefer their outside option.
        
        Args:
            init_u_i (torch.Tensor, optional): Initial bidder utilities. If None, uses default
                                             utilities (u_0 for all bidders). Defaults to None.
            init_mu_j (torch.Tensor, optional): Initial item assignments. If None, all items
                                              start unmatched. Defaults to None.
            eps (float, optional): Epsilon parameter for bid computation. Defaults to 0.
            return_mu_i_j (bool, optional): Whether to return binary assignment matrix.
                                          Defaults to False.
        
        Returns:
            tuple: (u_i, v_j, mu_j) or (u_i, v_j, mu_i_j) where:
                - u_i (torch.Tensor): Final bidder utilities of shape (num_i,)
                - v_j (torch.Tensor): Final item prices of shape (num_j,)
                - mu_j (torch.Tensor): Final item assignments of shape (num_j,)
                - mu_i_j (torch.Tensor): Binary assignment matrix of shape (num_i, num_j)
                                        (only if return_mu_i_j=True)
        
        Example:
            >>> # Run reverse auction
            >>> u_i, v_j, mu_j = auction.reverse_auction(eps=1e-4)
            >>> 
            >>> # Get binary assignment matrix
            >>> u_i, v_j, mu_i_j = auction.reverse_auction(eps=1e-4, return_mu_i_j=True)
        """
        u_i = self.init_u_i.clone() if init_u_i is None else init_u_i
        mu_j = self.init_mu_j.clone() if init_mu_j is None else init_mu_j
        unmatched_j = (mu_j == -1).nonzero(as_tuple=True)[0]

        while unmatched_j.numel() > 0:
            if self.method == "GS":
                unmatched_j = unmatched_j[:1]
            elif self.method == "batched":
                batch_size = max(1, int(self.sampling_rate * unmatched_j.size(0)))
                batch = unmatched_j[torch.randperm(unmatched_j.size(0))[:batch_size]]

            unmatched_j, u_i, mu_j = self._reverse_iteration(unmatched_j, u_i, mu_j, eps)

        v_j = self.get_V_i_j(u_i, j_id = self.all_j).amax(dim=0).clamp(min=self.v_0)

        if return_mu_i_j:
            mu_i_j = mu_j.unsqueeze(0) == self.all_i.unsqueeze(1)
            return u_i, v_j, mu_i_j

        return u_i, v_j, mu_j


    # Scaling method
    def forward_reverse_scaling(self, eps_init, eps_target, scaling_factor):
        """
        Run the forward-reverse auction with epsilon-scaling for improved convergence.
        
        This method alternates between forward and reverse auctions while gradually
        decreasing the epsilon parameter. This approach typically achieves better
        convergence properties than running either auction type alone.
        
        The algorithm starts with a large epsilon and gradually reduces it to the target
        value, using the solution from each phase as initialization for the next.
        
        Args:
            eps_init (float): Initial epsilon value (should be large)
            eps_target (float): Target epsilon value (final precision)
            scaling_factor (float): Factor by which to reduce epsilon each iteration
                                 (should be between 0 and 1)
        
        Returns:
            tuple: (u_i, v_j, mu_i_j) where:
                - u_i (torch.Tensor): Final bidder utilities of shape (num_i,)
                - v_j (torch.Tensor): Final item prices of shape (num_j,)
                - mu_i_j (torch.Tensor): Binary assignment matrix of shape (num_i, num_j)
        
        Example:
            >>> # Run with epsilon-scaling for better convergence
            >>> u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
            ...     eps_init=50, eps_target=1e-4, scaling_factor=0.5
            ... )
            >>> 
            >>> # Check equilibrium conditions
            >>> CS, feas, IR_i, IR_j = auction.check_equilibrium(u_i, v_j, mu_i_j, eps=1e-4)
        
        Note:
            This method is typically the recommended approach for solving ITU auction
            problems as it combines the strengths of both forward and reverse auctions
            while using epsilon-scaling to improve convergence speed and solution quality.
        """
        eps = eps_init
        v_j = self.init_v_j.clone()

        while True:
            u_i, v_j, mu_i  = self.forward_auction(init_v_j = v_j,  eps= eps)
            eps *=  scaling_factor
            u_i, v_j, mu_j  = self.reverse_auction(init_u_i = u_i, eps= eps)
            if eps <= eps_target:
                break

        u_i, v_j, mu_i_j  = self.reverse_auction(init_u_i = u_i, eps= eps, return_mu_i_j= True)
        mu_i = torch.where(mu_i_j.any(dim=1), mu_i_j.int().argmax(dim=1), -1)
        violations_i = (u_i > self.u_0) & (mu_i_j.sum(dim=1) == 0)
        mu_i[violations_i] = -1

        u_i, v_j, mu_i_j  = self.forward_auction(init_v_j = v_j, init_mu_i= mu_i, eps= eps, return_mu_i_j= True)
        # u_i, v_j, mu_i_j  = self.forward_auction(init_v_j = v_j, eps= eps, return_mu_i_j= True)


        return u_i, v_j, mu_i_j








