import torch
from itu_auction.core import ITUauction

def LU_template(Φ_i_j, α_i_j):
    """
    Create an ITU auction instance for the Linear Utility (LU) model.
    
    This template implements a model where utility transfer is linear but not perfect.
    The transfer rate α(i,j) determines how efficiently utility can be transferred
    between bidder i and item j. When α(i,j) = 1 for all pairs, this reduces to
    the TU model.
    
    The valuation functions are:
    - U(i,j) = Φ(i,j) - α(i,j) * v_j  (bidder utility = value - transfer_rate * price)
    - V(i,j) = (Φ(i,j) - u_i) / α(i,j)  (item value = (value - utility) / transfer_rate)
    
    where Φ(i,j) is the joint surplus and α(i,j) is the transfer rate.
    
    Args:
        Φ_i_j (torch.Tensor): Joint surplus matrix of shape (num_i, num_j)
                             where Φ_i_j[i,j] is the surplus from matching
                             bidder i to item j
        α_i_j (torch.Tensor): Transfer rate matrix of shape (num_i, num_j)
                             where α_i_j[i,j] is the efficiency of utility
                             transfer between bidder i and item j
                             (should be positive, typically between 0 and 1)
    
    Returns:
        ITUauction: Configured auction instance for the LU model
    
    Example:
        >>> import torch
        >>> from itu_auction.templates import LU_template
        >>> 
        >>> # Create random surplus and transfer rate matrices
        >>> num_i, num_j = 100, 98
        >>> Φ_i_j = torch.rand(num_i, num_j)
        >>> α_i_j = torch.rand(num_i, num_j) * 0.5 + 0.5  # Values between 0.5 and 1.0
        >>> 
        >>> # Create LU auction
        >>> auction = LU_template(Φ_i_j, α_i_j)
        >>> 
        >>> # Solve the assignment problem
        >>> u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
        ...     eps_init=50, eps_target=1e-4, scaling_factor=0.5
        ... )
    
    Note:
        The LU model generalizes the TU model by allowing imperfect utility transfer.
        Lower values of α(i,j) indicate less efficient utility transfer, making
        the assignment problem more complex. When α(i,j) = 1 for all pairs,
        this model reduces to the standard TU model.
    """
    
    def get_U_i_j(v_j, i_id, j_id=None):
        """
        Compute bidder utilities given item prices.
        
        Args:
            v_j (torch.Tensor): Item prices of shape (num_j,)
            i_id (torch.Tensor): Bidder indices
            j_id (torch.Tensor, optional): Item indices. If None, uses all items.
        
        Returns:
            torch.Tensor: Utility matrix of shape (len(i_id), num_j) or (len(i_id), len(j_id))
        """
        if j_id is None:
            return Φ_i_j[i_id] - α_i_j[i_id] * v_j.unsqueeze(0)
        else:
            return Φ_i_j[i_id, j_id] - α_i_j[i_id, j_id] * v_j

    def get_V_i_j(u_i, j_id, i_id=None):
        """
        Compute item values given bidder utilities.
        
        Args:
            u_i (torch.Tensor): Bidder utilities of shape (num_i,)
            j_id (torch.Tensor): Item indices
            i_id (torch.Tensor, optional): Bidder indices. If None, uses all bidders.
        
        Returns:
            torch.Tensor: Value matrix of shape (num_i, len(j_id)) or (len(i_id), len(j_id))
        """
        if i_id is None:
            return (Φ_i_j[:, j_id] - u_i.unsqueeze(1)) / α_i_j[:, j_id]
        else:
            return (Φ_i_j[i_id, j_id] - u_i) / α_i_j[i_id, j_id]

    num_i, num_j = Φ_i_j.shape
    return ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)