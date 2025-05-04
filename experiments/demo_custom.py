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
u_i, v_j, mu_i_j = TU_example.forward_auction(tol_ε= 1e-5)
CS, feas, IR_i, IR_j = TU_example.check_equilibrium(u_i, v_j, mu_i_j)

print("CS:", CS)
print("feas:", feas)
print("IR_i:", IR_i)
print("IR_j:", IR_j)


print(u_i.sum() + v_j.sum())
print((mu_i_j* Φ_i_j).sum())

