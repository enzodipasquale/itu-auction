import torch
from itu_auction.core import ITUauction

def convex_tax_template(α_i_j, γ_i_j, t_k, τ_k):
    """
    Create an ITU auction instance for the Convex Tax model.
    
    This template implements a model with convex taxes on utility transfers.
    The tax schedule is piecewise linear with rates τ_k at thresholds t_k.
    This creates a more complex utility transfer structure than the TU or LU models.
    
    The model parameters are:
    - α_i_j: Base utilities for each bidder-item pair
    - γ_i_j: Base values for each bidder-item pair  
    - t_k: Tax thresholds (increasing sequence)
    - τ_k: Tax rates at each threshold (between 0 and 1)
    
    The valuation functions compute utilities and values accounting for the
    convex tax structure on transfers.
    
    Args:
        α_i_j (torch.Tensor): Base utility matrix of shape (num_i, num_j)
                             where α_i_j[i,j] is the base utility for bidder i and item j
        γ_i_j (torch.Tensor): Base value matrix of shape (num_i, num_j)
                             where γ_i_j[i,j] is the base value for bidder i and item j
        t_k (torch.Tensor): Tax thresholds of shape (num_thresholds,)
                           should be in increasing order
        τ_k (torch.Tensor): Tax rates of shape (num_thresholds,)
                          should be between 0 and 1, typically increasing
    
    Returns:
        ITUauction: Configured auction instance for the convex tax model
    
    Example:
        >>> import torch
        >>> from itu_auction.templates import convex_tax_template
        >>> 
        >>> # Create model parameters
        >>> num_i, num_j = 100, 98
        >>> α_i_j = torch.rand(num_i, num_j)
        >>> γ_i_j = torch.rand(num_i, num_j)
        >>> 
        >>> # Create tax schedule (progressive tax)
        >>> t_k = torch.tensor([0.0, 1.0, 2.0, 5.0])  # Tax thresholds
        >>> τ_k = torch.tensor([0.0, 0.1, 0.2, 0.3])  # Tax rates
        >>> 
        >>> # Create convex tax auction
        >>> auction = convex_tax_template(α_i_j, γ_i_j, t_k, τ_k)
        >>> 
        >>> # Solve the assignment problem
        >>> u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
        ...     eps_init=50, eps_target=1e-4, scaling_factor=0.5
        ... )
    
    Note:
        The convex tax model introduces non-linearities in utility transfer,
        making the assignment problem more complex than standard models.
        The tax structure can capture various real-world constraints on
        utility transfers, such as transaction costs or regulatory limits.
    """
    
    # Precompute tax schedule parameters
    Δt = t_k[1:] - t_k[:-1]
    n_k = torch.cat([torch.zeros(1), torch.cumsum((1 - τ_k[:-1]) * Δt, dim=0)])
    N_k = n_k - (1 - τ_k) * t_k

    τ_k = τ_k.unsqueeze(0).clone()
    N_k = N_k.unsqueeze(0).clone()

    def get_U_i_j(v_j, i_id, j_id = None):
        """
        Compute bidder utilities given item prices with convex tax structure.
        
        Args:
            v_j (torch.Tensor): Item prices of shape (num_j,)
            i_id (torch.Tensor): Bidder indices
            j_id (torch.Tensor, optional): Item indices. If None, uses all items.
        
        Returns:
            torch.Tensor: Utility matrix accounting for convex taxes
        """
        if j_id is None:
            v_j = v_j.unsqueeze(0)
            return α_i_j[i_id] + (N_k.unsqueeze(0) + (1- τ_k.unsqueeze(0)) * (γ_i_j[i_id] - v_j).unsqueeze(2)).amin(-1)
        else:
            return α_i_j[i_id, j_id] + (N_k + (1- τ_k) * (γ_i_j[i_id, j_id] - v_j).unsqueeze(1)).amin(-1)

    def get_V_i_j(u_i, j_id, i_id = None):
        """
        Compute item values given bidder utilities with convex tax structure.
        
        Args:
            u_i (torch.Tensor): Bidder utilities of shape (num_i,)
            j_id (torch.Tensor): Item indices
            i_id (torch.Tensor, optional): Bidder indices. If None, uses all bidders.
        
        Returns:
            torch.Tensor: Value matrix accounting for convex taxes
        """
        if i_id is None:
            u_i = u_i.unsqueeze(1)
            return γ_i_j[:, j_id] + ((N_k.unsqueeze(0) + (α_i_j[:,j_id] - u_i).unsqueeze(2))/(1- τ_k.unsqueeze(0))).amin(-1)
        else:
            return γ_i_j[i_id,j_id] + ((N_k + (α_i_j[i_id,j_id] - u_i).unsqueeze(1))/(1- τ_k)).amin(-1)


    num_i, num_j = α_i_j.shape
    return ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)