from itu_auction.core import ITUauction
import torch

num_i = 5
num_j = 6
torch.manual_seed(0)
Φ_i_j = torch.randint(low=0, high= 2, size=(num_i, num_j))


def get_U_i_j(v_j, i_id, j_id=None):
    if j_id is None:
        return Φ_i_j[i_id] - v_j.unsqueeze(0)
    else:
        return Φ_i_j[i_id, j_id] - v_j

def get_V_i_j(u_i, j_id, i_id=None):
    if i_id is None:
        return Φ_i_j[:, j_id] - u_i.unsqueeze(0)
    else:
        return Φ_i_j[i_id, j_id] - u_i

TU_example = ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)
eps = 1e-3
u_i, v_j, mu_i_j = market.forward_auction(eps= eps)
CS, feas, IR_i, IR_j = market.check_equilibrium(u_i, v_j, mu_i_j, eps)

# Check duality
dual = u_i.sum() + v_j.sum()
primal = (mu_i_j * Φ_i_j).sum()

print("\n=== Duality Check ===")
print(f"Dual value                : {dual.item():.4f}")
print(f"Primal value (total cost) : {primal.item():.4f}")

