import torch
from itu_auction import get_template

# Example of using the TU template
num_i = 60
num_j = 50

torch.manual_seed(12)
Φ_i_j = (torch.randint(0, 10, (num_i, 1)) - torch.randint(0, 10, (1,num_j ))) **2
α_i_j = torch.randint(1, 10, (num_i, num_j))

market = get_template("LU")(Φ_i_j, α_i_j)

# Solve 
eps = 1e-3
# u_i, v_j, mu_i_j = market.forward_auction(eps= eps)
u_i, v_j, mu_i_j = market.reverse_auction(eps= eps)
# Check equilibrium
CS, feas, IR_i, IR_j = market.check_equilibrium(u_i, v_j, mu_i_j, eps)

print(u_i.sum() + v_j.sum())
print((mu_i_j* Φ_i_j).sum())