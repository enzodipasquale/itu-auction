import torch
from itu_auction import get_template

# Example of using the TU template
num_i = 60
num_j = 50

torch.manual_seed(123)
α_i_j = torch.randint(0, 5, (num_i, 1)) * torch.randint(0, 5, (1, num_j))
γ_i_j = torch.randint(0, 5, (num_i, 1)) * torch.randint(0, 5, (1, num_j))
t_k = torch.tensor([0, 9.701, 39.476, 84.201, 160.726, 204.101, 510.300]) / 100
τ_k = torch.tensor([.1, .12, .22, .24, .32, .35, .37])

market = get_template("convex_tax")(α_i_j, γ_i_j, t_k, τ_k)

# Solve 
u_i, v_j, mu_i_j = market.forward_auction(tol_ε= .01)

# Check equilibrium
CS, feas, IR_i, IR_j = market.check_equilibrium(u_i, v_j, mu_i_j, tol_ε= .01)


print(u_i.sum() + v_j.sum())
print((mu_i_j* (α_i_j + γ_i_j)).sum())