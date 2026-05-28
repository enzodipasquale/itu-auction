# ITU Auction Algorithms

An implementation of auction algorithms inspired by Bertsekas for solving assignment problems with non-separable valuations, i.e. with imperfectly transferable utility (ITU). The library uses **PyTorch** for parallel computation on CPU or GPU and supports batched problem solving.

---

## Overview

The package solves auction-based optimization problems where valuations may be non-separable across agents and items. Users supply their own valuation functions and the algorithm finds approximate equilibrium allocations using forward and reverse auction methods with epsilon-scaling.

---

## Features

- **User-defined valuation functions:** arbitrary valuation functions for bidders and items.
- **Batched problem solving:** solve multiple auction problems in parallel using PyTorch tensors and GPU acceleration.
- **Built-in valuation templates:**
  - Transferable utility (classic assignment problem)
  - Linear utility
  - Convex taxes
- **Auction methods:** forward and reverse auction procedures with convergence checks.
- **Epsilon-scaling:** improves convergence speed and solution quality.

---

## Installation

Install [PyTorch](https://pytorch.org/) following the instructions for your system and CUDA version, then install the package:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quick start

```python
import torch
from itu_auction import get_template

num_bidders, num_items = 100, 98
Φ_i_j = torch.rand(num_bidders, num_items)

auction = get_template("TU")(Φ_i_j)

u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)

CS, feas, IR_i, IR_j = auction.check_equilibrium(u_i, v_j, mu_i_j, eps=1e-4)
total_surplus = (mu_i_j * Φ_i_j).sum()
```

---

## Examples

### Transferable utility (TU)

```python
import torch
from itu_auction import get_template

Φ_i_j = torch.rand(50, 50)
auction = get_template("TU")(Φ_i_j)

# Forward auction
u_i, v_j, mu_i = auction.forward_auction(eps=1e-4)

# Reverse auction
u_i, v_j, mu_j = auction.reverse_auction(eps=1e-4)

# Combined with epsilon-scaling
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

### Linear utility (LU)

```python
import torch
from itu_auction import get_template

Φ_i_j = torch.rand(50, 50)
α_i_j = torch.rand(50, 50) * 0.5 + 0.5  # transfer rates in [0.5, 1.0]

auction = get_template("LU")(Φ_i_j, α_i_j)
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

### Convex tax

```python
import torch
from itu_auction import get_template

α_i_j = torch.rand(50, 50)
γ_i_j = torch.rand(50, 50)

t_k = torch.tensor([0.0, 1.0, 2.0, 5.0])  # tax thresholds
τ_k = torch.tensor([0.0, 0.1, 0.2, 0.3])  # tax rates

auction = get_template("convex_tax")(α_i_j, γ_i_j, t_k, τ_k)
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

### Custom valuation functions

```python
import torch
from itu_auction.core import ITUauction

def get_U_i_j(v_j, i_id):
    return Φ_i_j[i_id] - v_j.unsqueeze(0)

def get_V_i_j(u_i, j_id):
    return Φ_i_j[:, j_id] - u_i.unsqueeze(1)

auction = ITUauction(num_i=100, num_j=98, get_U_i_j=get_U_i_j, get_V_i_j=get_V_i_j)
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

### Batched processing

```python
import torch
from batched_itu_auction.core import ITUauction

def get_U_t_i_j(v_t_j, t_id, i_id):
    return Φ_t_i_j[t_id, i_id] - v_t_j[t_id]

def get_V_t_i_j(u_t_i, t_id, j_id):
    return Φ_t_i_j[t_id, :, j_id] - u_t_i[t_id]

auction = ITUauction(num_i=100, num_j=98, num_t=50,
                     get_U_t_i_j=get_U_t_i_j, get_V_t_i_j=get_V_t_i_j)
u_t_i, v_t_j, mu_t_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

---

## API

### `ITUauction` (single problem)

```python
ITUauction(num_i, num_j, get_U_i_j, get_V_i_j, lb=(0, 0))
```

- `num_i`, `num_j`: number of bidders and items
- `get_U_i_j`, `get_V_i_j`: bidder utility and item value functions
- `lb`: lower bounds for utilities and values

### `ITUauction` (batched)

```python
ITUauction(num_i, num_j, num_t, get_U_t_i_j, get_V_t_i_j, lb=(0, 0))
```

- `num_t`: number of parallel auction problems
- `get_U_t_i_j`, `get_V_t_i_j`: batched utility and value functions

### `get_template(name)`

Returns a pre-built auction class. Available names: `"TU"`, `"LU"`, `"convex_tax"`.

### `forward_reverse_scaling(eps_init, eps_target, scaling_factor)`

Solves the auction with epsilon-scaling. Returns `(u_i, v_j, mu_i_j)`: bidder utilities, item prices, and the binary assignment matrix.

### `check_equilibrium(u_i, v_j, mu_i_j, eps)`

Returns `(CS, feas, IR_i, IR_j)`: complementary slackness gap, feasibility flag, and individual rationality flags for bidders and items.

---

## Examples in the repo

The `experiments/` directory contains runnable examples:

- `demo_TU.py`
- `demo_LU.py`
- `demo_convex-tax.py`
- `demo_custom.py`
- `demo_batched.py`

```bash
python experiments/demo_TU.py
```

---

## License

MIT. See [LICENSE](LICENSE).
