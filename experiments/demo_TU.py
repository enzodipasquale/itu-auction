import torch
from itu_auction import get_template

# Example of using the TU template
num_i = 60
num_j = 50

torch.manual_seed(42)
Φ_i_j = (torch.randint(0, 9, (num_i, 1)) - torch.randint(0, 9, (1,num_j ))) **2
market = get_template("TU")(Φ_i_j)

# Solve 
u_i, v_j, mu_i_j = market.forward_auction(tol_ε= 1e-3)

# Check equilibrium
CS, feas, IR_i, IR_j = market.check_equilibrium(u_i, v_j, mu_i_j, tol_ε= 1e-3)

# Check duality
dual = u_i.sum() + v_j.sum()
primal = (mu_i_j * Φ_i_j).sum()

print("\n=== Duality Check ===")
print(f"Dual value                : {dual.item():.4f}")
print(f"Primal value (total cost) : {primal.item():.4f}")