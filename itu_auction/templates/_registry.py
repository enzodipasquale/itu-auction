from .TU import TU_template
from .LU import LU_template
from .convex_tax import convex_tax_template


TEMPLATE_REGISTRY = {
                    "TU": TU_template,
                    "LU": LU_template,  
                    "convex_tax": convex_tax_template,
                    }

def get_template(name):
    """
    Get a template function by name.
    
    This function provides access to pre-built auction templates that implement
    different utility transfer models. Each template creates an ITUauction instance
    with appropriate valuation functions for the specified model.
    
    Available templates:
    - "TU": Transferable Utility model (classic assignment problem)
    - "LU": Linear Utility model (imperfect utility transfer)
    - "convex_tax": Convex Tax model (non-linear utility transfer with taxes)
    
    Args:
        name (str): Name of the template to retrieve. Must be one of:
                   "TU", "LU", "convex_tax"
    
    Returns:
        callable: Template function that creates an ITUauction instance
        
    Raises:
        ValueError: If the template name is not recognized
    
    Example:
        >>> from itu_auction.templates import get_template
        >>> import torch
        >>> 
        >>> # Get TU template
        >>> TU_template = get_template("TU")
        >>> 
        >>> # Create surplus matrix
        >>> Φ_i_j = torch.rand(100, 98)
        >>> 
        >>> # Create auction instance
        >>> auction = TU_template(Φ_i_j)
        >>> 
        >>> # Solve the assignment problem
        >>> u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
        ...     eps_init=50, eps_target=1e-4, scaling_factor=0.5
        ... )
    
    Note:
        Templates provide a convenient way to create auction instances for common
        utility models. For custom models, you can create ITUauction instances
        directly with your own valuation functions.
    """
    try:
        return TEMPLATE_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown template name: {name}")
