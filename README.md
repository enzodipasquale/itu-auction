# ITU Auction Algorithms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **PyTorch-based implementation** of auction algorithms for solving assignment problems with **imperfectly transferable utility (ITU)**. This library provides efficient algorithms for finding approximate equilibrium allocations in markets with non-separable valuations, inspired by Bertsekas's auction theory.

## 🎯 Overview

This package implements sophisticated auction mechanisms for assignment problems where utility transfer between agents and items may be imperfect. Unlike traditional assignment problems with perfectly transferable utility, this framework handles complex market scenarios where:

- **Valuations are non-separable** across agents and items
- **Utility transfer is imperfect** (e.g., due to transaction costs, taxes, or constraints)
- **Market equilibrium** must satisfy complementary slackness, feasibility, and individual rationality

The algorithms use **epsilon-scaling** techniques to improve convergence and solution quality, making them suitable for both research and practical applications.

## ✨ Key Features

### 🔧 **Flexible Framework**
- **User-defined valuation functions** for arbitrary utility models
- **Template system** for common economic models (TU, LU, convex taxes)
- **Custom auction mechanisms** for specialized applications

### ⚡ **High Performance**
- **GPU acceleration** via PyTorch tensors
- **Batched processing** for multiple auction problems simultaneously
- **Parallel computation** for large-scale market simulations

### 🎲 **Auction Methods**
- **Forward auction**: Bidders compete for items
- **Reverse auction**: Items compete for bidders
- **Epsilon-scaling**: Improved convergence with gradual precision refinement
- **Equilibrium verification**: Built-in checks for market equilibrium conditions

### 📊 **Built-in Models**
- **Transferable Utility (TU)**: Classic assignment problem
- **Linear Utility (LU)**: Imperfect utility transfer with linear rates
- **Convex Taxes**: Non-linear utility transfer with tax schedules

## 🚀 Quick Start

### Installation

1. **Install PyTorch** (required dependency):
   ```bash
   # For CPU only
   pip install torch
   
   # For CUDA support (recommended)
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install ITU Auction**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Basic Usage

```python
import torch
from itu_auction import get_template

# Create a simple assignment problem
num_bidders, num_items = 100, 98
Φ_i_j = torch.rand(num_bidders, num_items)  # Surplus matrix

# Create auction instance using TU template
auction = get_template("TU")(Φ_i_j)

# Solve using epsilon-scaling (recommended)
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)

# Check equilibrium conditions
CS, feas, IR_i, IR_j = auction.check_equilibrium(u_i, v_j, mu_i_j, eps=1e-4)
print(f"Equilibrium satisfied: {CS <= 1e-4 and feas and IR_i and IR_j}")

# Compute total surplus
total_surplus = (mu_i_j * Φ_i_j).sum()
print(f"Total surplus: {total_surplus:.4f}")
```

## 📚 Detailed Examples

### Transferable Utility (TU) Model

```python
import torch
from itu_auction import get_template

# Create surplus matrix
Φ_i_j = torch.rand(50, 50)

# Create TU auction
auction = get_template("TU")(Φ_i_j)

# Solve with different methods
# Method 1: Forward auction only
u_i, v_j, mu_i = auction.forward_auction(eps=1e-4)

# Method 2: Reverse auction only  
u_i, v_j, mu_j = auction.reverse_auction(eps=1e-4)

# Method 3: Combined with epsilon-scaling (recommended)
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

### Linear Utility (LU) Model

```python
import torch
from itu_auction import get_template

# Create surplus and transfer rate matrices
Φ_i_j = torch.rand(50, 50)
α_i_j = torch.rand(50, 50) * 0.5 + 0.5  # Transfer rates between 0.5 and 1.0

# Create LU auction
auction = get_template("LU")(Φ_i_j, α_i_j)

# Solve the assignment problem
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

### Convex Tax Model

```python
import torch
from itu_auction import get_template

# Create model parameters
α_i_j = torch.rand(50, 50)  # Base utilities
γ_i_j = torch.rand(50, 50)  # Base values

# Create tax schedule (progressive tax)
t_k = torch.tensor([0.0, 1.0, 2.0, 5.0])  # Tax thresholds
τ_k = torch.tensor([0.0, 0.1, 0.2, 0.3])  # Tax rates

# Create convex tax auction
auction = get_template("convex_tax")(α_i_j, γ_i_j, t_k, τ_k)

# Solve the assignment problem
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

### Custom Valuation Functions

```python
import torch
from itu_auction.core import ITUauction

# Define custom valuation functions
def get_U_i_j(v_j, i_id):
    """Custom bidder utility function."""
    # Your custom logic here
    return Φ_i_j[i_id] - v_j.unsqueeze(0)

def get_V_i_j(u_i, j_id):
    """Custom item value function."""
    # Your custom logic here
    return Φ_i_j[:, j_id] - u_i.unsqueeze(1)

# Create custom auction instance
auction = ITUauction(num_i=100, num_j=98, get_U_i_j=get_U_i_j, get_V_i_j=get_V_i_j)

# Solve the auction
u_i, v_j, mu_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

### Batched Processing

```python
import torch
from batched_itu_auction.core import ITUauction

# Define batched valuation functions
def get_U_t_i_j(v_t_j, t_id, i_id):
    """Batched bidder utility function."""
    return Φ_t_i_j[t_id, i_id] - v_t_j[t_id]

def get_V_t_i_j(u_t_i, t_id, j_id):
    """Batched item value function."""
    return Φ_t_i_j[t_id, :, j_id] - u_t_i[t_id]

# Create batched auction instance
auction = ITUauction(num_i=100, num_j=98, num_t=50, 
                     get_U_t_i_j=get_U_t_i_j, get_V_t_i_j=get_V_t_i_j)

# Solve all problems simultaneously
u_t_i, v_t_j, mu_t_i_j = auction.forward_reverse_scaling(
    eps_init=50, eps_target=1e-4, scaling_factor=0.5
)
```

## 🔬 Advanced Usage

### Equilibrium Analysis

```python
# Check equilibrium conditions
CS, feas, IR_i, IR_j = auction.check_equilibrium(u_i, v_j, mu_i_j, eps=1e-4)

print("=== Equilibrium Analysis ===")
print(f"Complementary Slackness: {CS:.6f}")
print(f"Feasibility: {feas}")
print(f"Individual Rationality (Bidders): {IR_i}")
print(f"Individual Rationality (Items): {IR_j}")
print(f"Equilibrium Satisfied: {CS <= 1e-4 and feas and IR_i and IR_j}")
```

### Performance Optimization

```python
# Set device for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Φ_i_j = Φ_i_j.to(device)

# Configure auction parameters for better performance
auction.method = "batched"  # Use batched processing
auction.sampling_rate = 0.1  # Process 10% of unmatched agents per iteration
```

## 📖 API Reference

### Core Classes

#### `ITUauction`
Main auction algorithm class for single-problem solving.

```python
ITUauction(num_i, num_j, get_U_i_j, get_V_i_j, lb=(0, 0))
```

**Parameters:**
- `num_i` (int): Number of bidders/agents
- `num_j` (int): Number of items
- `get_U_i_j` (callable): Bidder utility function
- `get_V_i_j` (callable): Item value function
- `lb` (tuple): Lower bounds for utilities and values

#### `ITUauction` (Batched)
Batched version for parallel problem solving.

```python
ITUauction(num_i, num_j, num_t, get_U_t_i_j, get_V_t_i_j, lb=(0, 0))
```

**Additional Parameters:**
- `num_t` (int): Number of parallel auction problems
- `get_U_t_i_j` (callable): Batched bidder utility function
- `get_V_t_i_j` (callable): Batched item value function

### Template Functions

#### `get_template(name)`
Get pre-built auction templates.

**Available templates:**
- `"TU"`: Transferable Utility model
- `"LU"`: Linear Utility model  
- `"convex_tax"`: Convex Tax model

### Main Methods

#### `forward_reverse_scaling(eps_init, eps_target, scaling_factor)`
Recommended method for solving auction problems.

**Parameters:**
- `eps_init` (float): Initial epsilon value (typically 50)
- `eps_target` (float): Target epsilon value (typically 1e-4)
- `scaling_factor` (float): Epsilon reduction factor (typically 0.5)

**Returns:**
- `u_i` (torch.Tensor): Final bidder utilities
- `v_j` (torch.Tensor): Final item prices
- `mu_i_j` (torch.Tensor): Binary assignment matrix

## 🧪 Examples and Experiments

The `experiments/` directory contains ready-to-run examples:

- `demo_TU.py`: Transferable utility examples
- `demo_LU.py`: Linear utility examples
- `demo_convex-tax.py`: Convex tax examples
- `demo_custom.py`: Custom valuation function examples
- `demo_batched.py`: Batched processing examples

Run any example with:
```bash
python experiments/demo_TU.py
```

## 🔧 Configuration

### Device Selection
```python
# Automatic device selection (recommended)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Manual device selection
device = torch.device("cuda:0")  # Specific GPU
device = torch.device("cpu")     # CPU only
```

### Epsilon Parameters
```python
# Conservative settings (slower, more accurate)
eps_init = 100
eps_target = 1e-6
scaling_factor = 0.3

# Aggressive settings (faster, less accurate)
eps_init = 10
eps_target = 1e-3
scaling_factor = 0.7
```

## 📊 Performance Tips

1. **Use GPU acceleration** when available for significant speedup
2. **Batch multiple problems** for parallel processing
3. **Adjust epsilon parameters** based on accuracy vs. speed requirements
4. **Use appropriate templates** for your specific utility model
5. **Monitor convergence** with equilibrium checks

## 🤝 Contributing

We welcome contributions! Please feel free to:

- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Bertsekas's auction theory
- Built with PyTorch for efficient computation
- Designed for research and practical applications in market design

---

**For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/enzodipasquale/itu-auction).**
