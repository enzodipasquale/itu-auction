"""
Experiment 3: directly measure ||v - v*||_inf between two epsilon-CS profiles
on the SAME ITU instance, to test the price-stability lemma.

Two ways to generate distinct epsilon-CS profiles on the same instance:
  (A) forward-only auction (consumer-optimal-ish)  vs.  reverse-only auction (yogurt-optimal-ish).
      Different sides typically pick different points in the equilibrium lattice.
  (B) randomize bidder order (in batched/sampling mode).

We use (A): contrast forward-only and reverse-only outputs.

The price-stability lemma we conjecture says: ||v_fwd - v_rev||_inf <= O(N * eps).

If empirically the ratio (||v_fwd - v_rev||_inf) / eps grows like O(N) and is
INDEPENDENT of kappa, that's strong evidence for the conjecture. If it grows
faster than O(N) (e.g. with kappa) or with 1/eps, the conjecture fails.
"""
import sys, os, statistics, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrumented import make_TU, make_LU
import torch


def generate_two_profiles(market, eps):
    """Run forward-only and reverse-only at fixed eps; return both v profiles."""
    # Forward-only at fixed eps — produces ε-CS w.r.t. consumers
    u1, v1, mu1 = market.forward_auction(eps=eps, return_mu_i_j=True)
    # Reverse-only at fixed eps — produces ε-CS w.r.t. yogurts
    u2, v2, mu2 = market.reverse_auction(eps=eps, return_mu_i_j=True)
    return v1.detach().clone(), v2.detach().clone(), u1, u2, mu1, mu2


def check_eps_cs(market, u, v, eps):
    """Verify (eps-CS): u_i >= U_ij(v_j) - eps for all (i,j)."""
    U = market.get_U_i_j(v, market.all_i)  # shape (m, n)
    violation = (U - u.unsqueeze(1) - eps).clamp(min=0).max().item()
    return violation


def main():
    eps_targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    Ns = [20, 50, 100]
    kappas = [1, 10, 100, 1000]
    seeds = [1, 2, 3, 4]

    print("# Direct test of price-stability lemma")
    print("# Measure ||v_fwd - v_rev||_inf  on the same ITU instance.")
    print("# If the price-stability lemma holds, this should scale as O(N * eps).\n")

    print(f"{'kappa':>6s} {'N':>4s} {'eps':>9s} {'||v_f-v_r||_inf':>18s} "
          f"{'norm/eps':>10s} {'norm/(N*eps)':>14s} {'CS_fwd':>10s} {'CS_rev':>10s}")
    rows = []
    for kappa in kappas:
        for N in Ns:
            for eps in eps_targets:
                diffs, ratios_eps, ratios_neps = [], [], []
                cs_fwd_max, cs_rev_max = 0.0, 0.0
                for s in seeds:
                    if kappa == 1:
                        m, _ = make_TU(N, N, seed=s)
                    else:
                        m, _, _, _ = make_LU(N, N, seed=s, beta_min=1.0, beta_max=kappa)
                    v1, v2, u1, u2, _, _ = generate_two_profiles(m, eps)
                    diff = (v1 - v2).abs().max().item()
                    diffs.append(diff)
                    ratios_eps.append(diff / eps)
                    ratios_neps.append(diff / (N * eps))
                    cs_fwd_max = max(cs_fwd_max, check_eps_cs(m, u1, v1, eps))
                    cs_rev_max = max(cs_rev_max, check_eps_cs(m, u2, v2, eps))
                d_avg = statistics.mean(diffs)
                r_eps = statistics.mean(ratios_eps)
                r_neps = statistics.mean(ratios_neps)
                rows.append((kappa, N, eps, d_avg, r_eps, r_neps))
                print(f"{kappa:>6d} {N:>4d} {eps:>9.1e} {d_avg:>18.4e} "
                      f"{r_eps:>10.2f} {r_neps:>14.4f} {cs_fwd_max:>10.2e} {cs_rev_max:>10.2e}")
            print()

    # Hypothesis tests
    print("\n# Scaling analysis: is ||v_fwd-v_rev||_inf <= C * N * eps for some C uniform in kappa?")
    print("# Equivalently, does ratio / (N*eps) stay bounded as eps -> 0 and as kappa grows?\n")
    print(f"{'kappa':>6s} {'N':>4s} {'max ratio/(N*eps)':>20s} {'mean ratio/(N*eps)':>22s}")
    bykN = {}
    for kappa, N, eps, d, r_eps, r_neps in rows:
        bykN.setdefault((kappa, N), []).append(r_neps)
    for (kappa, N), rs in sorted(bykN.items()):
        print(f"{kappa:>6d} {N:>4d} {max(rs):>20.4f} {statistics.mean(rs):>22.4f}")


if __name__ == "__main__":
    main()
