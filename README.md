# ITU Auction Algorithms

An implementation of auction algorithms inspired by Bertsekas for solving assignment problems with non-separable valuations. This library leverages **PyTorch** for efficient parallel computation and supports batched problem solving.

---

## Overview

This package provides a flexible framework to solve auction-based optimization problems where valuations may be non-separable across agents and items. Users supply their own valuation functions, and the algorithm finds approximate equilibrium allocations efficiently using forward and reverse auction methods.

---

## Features

- **User-defined valuation functions:** Input arbitrary valuation functions for bidders and items.
- **Batched problem solving:** Solve multiple auction problems in parallel using PyTorch tensors and GPU acceleration.
- **Built-in valuation functions:** Includes standard models such as:
  - Transferable utility (classic assignment problem)
  - Linear utility
  - Convex taxes
- **Auction methods:** Implements both forward and reverse auction procedures with convergence checks.
- **Scaling method:** Supports epsilon-scaling to improve convergence speed and solution quality.
- **PyTorch-based:** Fully utilizes GPU or CPU parallelism for speed and scalability.

---
