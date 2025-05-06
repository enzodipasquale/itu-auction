import torch
from itu_auction import get_template
import time

# Example of using the TU template
num_i = 60
num_j = 70

torch.manual_seed(42)
α_i_j = torch.randint(0, 5, (num_i, 1)) * torch.randint(0, 5, (1, num_j))
γ_i_j = torch.randint(0, 5, (num_i, 1)) * torch.randint(0, 5, (1, num_j))
t_k = torch.tensor([0, 9.701, 39.476, 84.201, 160.726, 204.101, 510.300]) / 100
τ_k = torch.tensor([.1, .12, .22, .24, .32, .35, .37])

market = get_template("convex_tax")(α_i_j, γ_i_j, t_k, τ_k)

# Solve 
eps = 1e-3
scaling_factor = 0.5
eps_init = 10
tic = time.time()
u_i, v_j, mu_i_j = market.forward_auction(eps= eps)
# u_i, v_j, mu_i_j = market.reverse_auction(eps= eps)
# u_i, v_j, mu_i_j = market.forward_reverse_scaling(eps_init, eps, scaling_factor)
toc = time.time()
print(f"Time taken for auction: {toc - tic:.4f} seconds")
# Check equilibrium
CS, feas, IR_j = market.check_equilibrium(u_i, v_j, mu_i_j, eps)

# Check duality
Φ_i_j = α_i_j + γ_i_j
dual = u_i.sum() + v_j.sum()
primal = (mu_i_j * Φ_i_j).sum()

print("\n=== Duality Check ===")
print(f"Dual value                : {dual.item():.4f}")
print(f"Primal value (total cost) : {primal.item():.4f}")