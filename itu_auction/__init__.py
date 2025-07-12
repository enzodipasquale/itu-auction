"""
ITU Auction Algorithms Package

A PyTorch-based implementation of auction algorithms for solving assignment problems
with imperfectly transferable utility (ITU). This package provides efficient algorithms
for finding approximate equilibrium allocations in markets with non-separable valuations.

The package supports:
- Forward and reverse auction procedures
- Epsilon-scaling for improved convergence
- Batched processing for multiple problems
- GPU acceleration via PyTorch
- Built-in templates for common utility models

Main Components:
- ITUauction: Core auction algorithm class
- Templates: Pre-built valuation functions for common models
- Batched processing: Parallel solution of multiple problems

Example Usage:
    >>> from itu_auction import get_template
    >>> import torch
    >>> 
    >>> # Create a transferable utility auction
    >>> Φ_i_j = torch.rand(100, 98)  # Surplus matrix
    >>> auction = get_template("TU")(Φ_i_j)
    >>> 
    >>> # Solve using epsilon-scaling
    >>> u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    ...     eps_init=50, eps_target=1e-4, scaling_factor=0.5
    ... )
    >>> 
    >>> # Check equilibrium conditions
    >>> CS, feas, IR_i, IR_j = auction.check_equilibrium(u_i, v_j, mu_i_j, eps=1e-4)

For more information, see the README.md file or individual module docstrings.
"""

from .templates import get_template
from .core import ITUauction

__version__ = "0.1"
__author__ = "Enzo Di Pasquale"

__all__ = ["ITUauction", "get_template"] 