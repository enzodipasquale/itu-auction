import os

import torch
import torch.distributed as dist

from distributed_itu_auction import (
    DistributedConfig,
    get_distributed_template,
    shard_bounds,
)


backend = "nccl" if torch.cuda.is_available() else "gloo"
if backend == "nccl":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend)
rank = dist.get_rank()
world_size = dist.get_world_size()

config = DistributedConfig(2, 2, 2 * world_size, backend=backend)
start, stop = shard_bounds(config.num_t, rank, world_size)
base = torch.tensor([[4.0, 1.0], [0.0, 3.0]])
phi = torch.stack(
    [torch.roll(base, shifts=t, dims=1) for t in range(start, stop)]
)

auction = get_distributed_template("TU")(phi, config)
u, v, matching = auction.forward_reverse_scaling(1.0, 0.25, 0.5)
cs, feasible, ir_i, ir_j = auction.check_equilibrium(u, v, matching, eps=0.25)

if rank == 0:
    print(
        "{} ranks, {} markets, CS={:.3g}, feasible={}, IR=({}, {})".format(
            world_size,
            config.num_t,
            cs,
            feasible,
            ir_i,
            ir_j,
        )
    )

dist.destroy_process_group()
