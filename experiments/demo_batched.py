from time import perf_counter

import torch

from batched_itu_auction import ITUauction as BatchedAuction


def main():
    torch.manual_seed(29)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_t, num_i, num_j = 16, 32, 28
    surplus = torch.rand(num_t, num_i, num_j, device=device)

    def get_u(v, t, i, j=None):
        if j is None:
            return surplus[t, i] - v[t]
        return surplus[t, i, j] - v

    def get_v(u, t, j, i=None):
        if i is None:
            if t.ndim > 1:
                t = t.reshape(-1)
                j = j.reshape(-1)
                return surplus[t][:, :, j] - u[t].unsqueeze(2)
            return surplus[t, :, j] - u[t]
        return surplus[t, i, j] - u

    auction = BatchedAuction(num_i, num_j, num_t, get_u, get_v, device=device)
    eps = 1e-3

    started = perf_counter()
    u, v, matching = auction.forward_reverse_scaling(1.0, eps, 0.5)
    elapsed = perf_counter() - started

    cs, feasible, ir_i, ir_j = auction.check_equilibrium(u, v, matching, eps)
    primal = (matching * surplus).sum((1, 2))
    dual = u.sum(1) + v.sum(1)

    print(f"device={device} markets={num_t} elapsed={elapsed:.3f}s")
    print(
        f"max_gap={(dual - primal).amax().item():.3g} CS={cs.item():.3g} "
        f"feasible={bool(feasible)} IR=({bool(ir_i)}, {bool(ir_j)})"
    )


if __name__ == "__main__":
    main()
