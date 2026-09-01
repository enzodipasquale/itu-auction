# ITU auction algorithms

PyTorch implementations of forward and reverse auction algorithms for one-to-one matching with imperfectly transferable utility. The repository supports a single market, batches of independent markets on one device, and batches sharded across `torch.distributed` ranks.

The economic setting follows Bonnet, Galichon, Hsieh, O'Hara, and Shum, [*Yogurts Choose Consumers? Estimation of Random-Utility Models via Two-Sided Matching*](https://arxiv.org/abs/2111.13744).

## Install

Python 3.8 or later and PyTorch 2.0 or later are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

Run the tests with:

```bash
python -m pytest -q
```

## Single market

```python
import torch

from itu_auction import get_template

surplus = torch.rand(100, 98)
auction = get_template("TU")(surplus)

u, v, matching = auction.forward_reverse_scaling(
    eps_init=1.0,
    eps_target=1e-4,
    scaling_factor=0.5,
    certify_terminal=True,
)

cs, feasible, ir_i, ir_j = auction.check_equilibrium(
    u, v, matching, eps=1e-4
)
```

`matching` is a Boolean tensor with shape `(num_i, num_j)`. The solver also exposes `forward_auction` and `reverse_auction` for fixed-epsilon runs. Both sides of a market must contain at least two alternatives.

Three templates are included:

| Name | Constructor arguments | Utility |
| --- | --- | --- |
| `TU` | `surplus` | `U_ij(v) = surplus_ij - v` |
| `LU` | `surplus, transfer_rate` | `U_ij(v) = surplus_ij - transfer_rate_ij v` |
| `convex_tax` | `alpha, gamma, thresholds, rates` | Piecewise-linear after-tax transfers |

For another ITU specification, construct `itu_auction.core.ITUauction` with a pair of inverse utility functions. See [`experiments/demo_custom.py`](experiments/demo_custom.py) for a complete example.

## Batched and distributed runs

| Mode | Work placement | Entry point |
| --- | --- | --- |
| Single | One market on one device | `itu_auction.ITUauction` |
| Batched | Independent markets on one device | `batched_itu_auction.ITUauction` |
| Distributed | Contiguous market shards across ranks | `distributed_itu_auction.DistributedITUAuction` |

The batched solver keeps a leading market dimension and performs assignment reductions across all active markets on the selected device. [`experiments/demo_batched.py`](experiments/demo_batched.py) shows the expected tensor shapes for custom batched utility functions.

The distributed solver partitions only that leading dimension. Auction iterations remain local to each rank; collective operations are limited to global equilibrium checks and optional result gathering. The distributed template currently covers transferable utility. It accepts either the full surplus tensor on every rank or the rank's contiguous shard.

Run the two-rank smoke test locally with:

```bash
torchrun --standalone --nproc-per-node=2 experiments/demo_distributed.py
```

On Slurm, the included launcher requests one task per GPU and a five-minute allocation:

```bash
sbatch --account=YOUR_ACCOUNT experiments/demo_distributed_slurm.sh
```

Set `ITU_PYTHON` when the cluster environment does not expose the intended interpreter as `python`.

## Examples

All examples are deterministic and sized to run quickly:

```bash
python experiments/demo_TU.py
python experiments/demo_LU.py
python experiments/demo_convex-tax.py
python experiments/demo_custom.py
python experiments/demo_batched.py
```

## License

MIT
