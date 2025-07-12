import torch
from itu_auction.core import ITUauction

def TU_template(Φ_i_j):
    """
    Create an ITU auction instance for the Transferable Utility (TU) model.
    
    This template implements the classic assignment problem where utility is perfectly
    transferable between agents and items. In this model, the total surplus from
    an assignment can be freely redistributed between the matched parties.
    
    The valuation functions are:
    - U(i,j) = Φ(i,j) - v_j  (bidder utility = value - price)
    - V(i,j) = Φ(i,j) - u_i  (item value = value - utility)
    
    where Φ(i,j) is the joint surplus from matching bidder i to item j.
    
    Args:
        Φ_i_j (torch.Tensor): Joint surplus matrix of shape (num_i, num_j)
                             where Φ_i_j[i,j] is the surplus from matching
                             bidder i to item j
    
    Returns:
        ITUauction: Configured auction instance for the TU model
    
    Example:
        >>> import torch
        >>> from itu_auction.templates import TU_template
        >>> 
        >>> # Create random surplus matrix
        >>> num_i, num_j = 100, 98
        >>> Φ_i_j = torch.rand(num_i, num_j)
        >>> 
        >>> # Create TU auction
        >>> auction = TU_template(Φ_i_j)
        >>> 
        >>> # Solve the assignment problem
        >>> u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
        ...     eps_init=50, eps_target=1e-4, scaling_factor=0.5
        ... )
        >>> 
        >>> # Check total surplus
        >>> total_surplus = (mu_i_j * Φ_i_j).sum()
        >>> print(f"Total surplus: {total_surplus:.4f}")
    
    Note:
        This is the standard model for assignment problems and is equivalent to
        the Hungarian algorithm when solved optimally. The auction algorithm
        provides an approximate solution that converges to the optimal solution
        as epsilon approaches zero.
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
            return Φ_i_j[i_id] - v_j.unsqueeze(0)
        else:
            return Φ_i_j[i_id, j_id] - v_j

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
            return Φ_i_j[:, j_id] - u_i.unsqueeze(1)
        else:
            return Φ_i_j[i_id, j_id] - u_i

    num_i, num_j = Φ_i_j.shape
    return ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)
