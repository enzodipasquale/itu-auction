"""Final numerical runs for the ITU auction dissertation chapter.

This script is deliberately self-contained: it uses the same instrumented
auction wrapper as the pilot logs, adds a simple consumer-specific-additive
(CSA) generator, and writes run-level plus table-level CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from instrumented import CountingAuction, make_LU, make_TU


DEFAULT_BASELINE_EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
DEFAULT_ROBUST_EPS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def make_csa_linear(num_i, num_j, seed=0, scale=10.0, slope_min=1.0, slope_max=10.0):
    """Build a CSA instance U_ij(v) = a_i * (alpha_ij - v).

    This is a structured ITU class covered by the consumer-specific additivity
    theorem: F_i(z) = a_i z. It is intentionally simple so the inverse is exact
    and the experiment isolates the algorithmic class rather than numerical
    root-finding.
    """

    torch.manual_seed(seed)
    alpha = torch.rand(num_i, num_j) * scale
    slopes = torch.rand(num_i) * (slope_max - slope_min) + slope_min

    def get_U_i_j(v_j, i_id, j_id=None):
        if j_id is None:
            return slopes[i_id].unsqueeze(1) * (alpha[i_id] - v_j.unsqueeze(0))
        return slopes[i_id] * (alpha[i_id, j_id] - v_j)

    def get_V_i_j(u_i, j_id, i_id=None):
        if i_id is None:
            return alpha[:, j_id] - u_i.unsqueeze(1) / slopes.unsqueeze(1)
        return alpha[i_id, j_id] - u_i / slopes[i_id]

    kappa = float(slopes.max() / slopes.min())
    return CountingAuction(num_i, num_j, get_U_i_j, get_V_i_j), alpha, slopes, kappa


def summarize(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def silent_cs_violation(market, u_i, v_j):
    with torch.no_grad():
        return float((market.get_U_i_j(v_j, market.all_i).amax(dim=1) - u_i).amax().item())


def run_market(market, eps_target, eps_init, theta):
    market._reset_counters()
    start = time.perf_counter()
    u_i, v_j, mu_i_j = market.forward_reverse_scaling(eps_init, eps_target, theta)
    elapsed = time.perf_counter() - start
    return {
        "total_iters": market.total_iters,
        "fwd_phases": len(market.fwd_iters_per_phase),
        "rev_phases": len(market.rev_iters_per_phase),
        "runtime_sec": elapsed,
        "cs_violation": silent_cs_violation(market, u_i, v_j),
        "matched_pairs": int(mu_i_j.sum().item()),
    }


def run_case(case, eps_grid, seeds, eps_init, theta):
    rows = []
    for eps_target in eps_grid:
        for seed in seeds:
            print(f"  {case['name']} eps={eps_target:.0e} seed={seed}", flush=True)
            market, actual_kappa = case["maker"](seed)
            result = run_market(market, eps_target, eps_init, theta)
            rows.append(
                {
                    "experiment": case["experiment"],
                    "case": case["name"],
                    "instance_class": case["instance_class"],
                    "N": case["N"],
                    "target_kappa": case["target_kappa"],
                    "actual_kappa": actual_kappa,
                    "eps_target": eps_target,
                    "seed": seed,
                    **result,
                }
            )
    return rows


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = (
            row["experiment"],
            row["case"],
            row["instance_class"],
            row["N"],
            row["target_kappa"],
            row["eps_target"],
        )
        groups.setdefault(key, []).append(row)

    out = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        experiment, case, instance_class, n, target_kappa, eps_target = key
        iter_mean, iter_sd = summarize([g["total_iters"] for g in group])
        time_mean, time_sd = summarize([g["runtime_sec"] for g in group])
        kappa_mean, kappa_sd = summarize([g["actual_kappa"] for g in group])
        cs_max = max(g["cs_violation"] for g in group)
        out.append(
            {
                "experiment": experiment,
                "case": case,
                "instance_class": instance_class,
                "N": n,
                "target_kappa": target_kappa,
                "eps_target": eps_target,
                "seeds": len(group),
                "actual_kappa_mean": kappa_mean,
                "actual_kappa_sd": kappa_sd,
                "iterations_mean": iter_mean,
                "iterations_sd": iter_sd,
                "runtime_sec_mean": time_mean,
                "runtime_sec_sd": time_sd,
                "fwd_phases": group[0]["fwd_phases"],
                "rev_phases": group[0]["rev_phases"],
                "cs_violation_max": cs_max,
            }
        )
    return out


def compact_table_rows(agg_rows):
    by_case = {}
    for row in agg_rows:
        key = (
            row["experiment"],
            row["case"],
            row["instance_class"],
            row["N"],
            row["target_kappa"],
        )
        by_case.setdefault(key, {})[row["eps_target"]] = row

    compact = []
    for key, eps_rows in sorted(by_case.items()):
        experiment, case, instance_class, n, target_kappa = key
        eps_min = min(eps_rows)
        eps_max = max(eps_rows)
        row_min = eps_rows[eps_min]
        row_max = eps_rows[eps_max]
        compact.append(
            {
                "experiment": experiment,
                "case": case,
                "instance_class": instance_class,
                "N": n,
                "target_kappa": target_kappa,
                "eps_high": eps_max,
                "iterations_at_eps_high": row_max["iterations_mean"],
                "runtime_at_eps_high": row_max["runtime_sec_mean"],
                "eps_low": eps_min,
                "iterations_at_eps_low": row_min["iterations_mean"],
                "runtime_at_eps_low": row_min["runtime_sec_mean"],
                "seeds": row_min["seeds"],
            }
        )
    return compact


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, agg_rows, compact_rows, metadata):
    lines = []
    lines.append("# Final chapter experiment summary")
    lines.append("")
    lines.append("This file is generated by `final_chapter_experiments.py`.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Compact Table Rows")
    lines.append("")
    lines.append("| Experiment | Case | N | Target kappa | eps high iter/time | eps low iter/time | Seeds |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in compact_rows:
        lines.append(
            "| {experiment} | {case} | {N} | {target_kappa} | "
            "{iterations_at_eps_high:.1f} / {runtime_at_eps_high:.4f}s | "
            "{iterations_at_eps_low:.1f} / {runtime_at_eps_low:.4f}s | {seeds} |".format(**row)
        )
    lines.append("")
    lines.append("## Full Aggregates")
    lines.append("")
    lines.append("| Experiment | Case | eps | Iter mean | Iter sd | Runtime mean | Runtime sd | CS max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in agg_rows:
        lines.append(
            "| {experiment} | {case} | {eps_target:.0e} | "
            "{iterations_mean:.1f} | {iterations_sd:.1f} | "
            "{runtime_sec_mean:.4f} | {runtime_sec_sd:.4f} | {cs_violation_max:.2e} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def build_cases():
    baseline_n = 50
    return [
        {
            "experiment": "baseline",
            "name": "TU",
            "instance_class": "TU",
            "N": baseline_n,
            "target_kappa": 1,
            "maker": lambda seed: (lambda out: (out[0], 1.0))(make_TU(baseline_n, baseline_n, seed=seed)),
        },
        {
            "experiment": "baseline",
            "name": "CSA-linear(kappa=2)",
            "instance_class": "CSA-linear",
            "N": baseline_n,
            "target_kappa": 2,
            "maker": lambda seed: (
                lambda out: (out[0], out[3])
            )(make_csa_linear(baseline_n, baseline_n, seed=seed, slope_min=1.0, slope_max=2.0)),
        },
        {
            "experiment": "baseline",
            "name": "LTU(kappa=10)",
            "instance_class": "LTU",
            "N": baseline_n,
            "target_kappa": 10,
            "maker": lambda seed: (
                lambda out: (out[0], out[3])
            )(make_LU(baseline_n, baseline_n, seed=seed, beta_min=1.0, beta_max=10.0)),
        },
        *[
            {
                "experiment": "robustness",
                "name": f"LTU(N={n},kappa={kappa})",
                "instance_class": "LTU",
                "N": n,
                "target_kappa": kappa,
                "maker": lambda seed, n=n, kappa=kappa: (
                    lambda out: (out[0], out[3])
                )(make_LU(n, n, seed=seed, beta_min=1.0, beta_max=float(kappa))),
            }
            for n in [50, 100, 200]
            for kappa in [1000, 10000]
        ],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--eps-init", type=float, default=50.0)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--out-dir", default="experiments/research_log_eps/final_outputs")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run two seeds and shortened grids.")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    baseline_eps = DEFAULT_BASELINE_EPS
    robust_eps = DEFAULT_ROBUST_EPS
    if args.smoke:
        seeds = seeds[:2]
        baseline_eps = [1e-2, 1e-4]
        robust_eps = [1e-2, 1e-4]

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for case in build_cases():
        eps_grid = baseline_eps if case["experiment"] == "baseline" else robust_eps
        print(
            f"Running {case['name']} over {len(eps_grid)} eps values and {len(seeds)} seeds",
            flush=True,
        )
        all_rows.extend(run_case(case, eps_grid, seeds, args.eps_init, args.theta))

    agg_rows = aggregate_rows(all_rows)
    compact_rows = compact_table_rows(agg_rows)

    metadata = {
        "run_tag": tag,
        "run_datetime": datetime.now().isoformat(timespec="seconds"),
        "seeds": f"{seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)",
        "eps_init": args.eps_init,
        "theta": args.theta,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "torch": torch.__version__,
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    raw_path = out_dir / f"final_chapter_raw_{tag}.csv"
    agg_path = out_dir / f"final_chapter_aggregates_{tag}.csv"
    compact_path = out_dir / f"final_chapter_compact_{tag}.csv"
    meta_path = out_dir / f"final_chapter_metadata_{tag}.json"
    md_path = out_dir / f"final_chapter_summary_{tag}.md"

    write_csv(raw_path, all_rows)
    write_csv(agg_path, agg_rows)
    write_csv(compact_path, compact_rows)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    write_markdown(md_path, agg_rows, compact_rows, metadata)

    print(f"Wrote {raw_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {compact_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
