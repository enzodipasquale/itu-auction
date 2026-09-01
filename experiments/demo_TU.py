from time import perf_counter

import torch

from itu_auction import get_template


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surplus = torch.rand(48, 40, device=device)
    auction = get_template("TU")(surplus)

    eps = 1e-3
    started = perf_counter()
    u, v, matching = auction.forward_reverse_scaling(
        1.0, eps, 0.5, certify_terminal=True
    )
    elapsed = perf_counter() - started

    cs, feasible, ir_i, ir_j = auction.check_equilibrium(u, v, matching, eps)
    primal = (matching * surplus).sum()
    dual = u.sum() + v.sum()

    print(f"device={device} elapsed={elapsed:.3f}s")
    print(
        f"surplus={primal.item():.6f} gap={(dual - primal).item():.3g} "
        f"CS={cs.item():.3g} feasible={bool(feasible)} IR=({bool(ir_i)}, {bool(ir_j)})"
    )


if __name__ == "__main__":
    main()
