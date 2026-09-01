import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from batched_itu_auction import ITUauction as BatchedITUAuction
from distributed_itu_auction import (
    DistributedConfig,
    get_distributed_template,
    shard_bounds,
)
from distributed_itu_auction.core import _backend, _process_group


def _surplus(start, stop, num_i, num_j):
    markets = []
    for t in range(start, stop):
        generator = torch.Generator().manual_seed(1729 + t)
        markets.append(torch.rand((num_i, num_j), generator=generator))
    return torch.stack(markets)


def _distributed_worker(rank, world_size, init_method):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    try:
        config = DistributedConfig(4, 3, 5, backend="gloo", device="cpu")
        start, stop = shard_bounds(config.num_t, rank, world_size)
        phi = _surplus(start, stop, config.num_i, config.num_j)
        auction = get_distributed_template("TU")(phi, config)
        local = auction.forward_reverse_scaling(2.0, 0.125, 0.5)
        gathered = auction.gather_results(*local)
        cs, feasible, ir_i, ir_j = auction.check_equilibrium(*local, eps=0.125)

        assert cs <= 0.125 + 1e-6
        assert feasible and ir_i and ir_j

        if rank == 0:
            phi = _surplus(0, config.num_t, config.num_i, config.num_j)

            def get_u(v, t, i, j=None):
                if j is None:
                    return phi[t, i] - v[t]
                return phi[t, i, j] - v

            def get_v(u, t, j, i=None):
                if i is None:
                    if t.ndim > 1:
                        t = t.reshape(-1)
                        j = j.reshape(-1)
                        return phi[t][:, :, j] - u[t].unsqueeze(2)
                    return phi[t, :, j] - u[t]
                return phi[t, i, j] - u

            reference = BatchedITUAuction(
                config.num_i,
                config.num_j,
                config.num_t,
                get_u,
                get_v,
                device="cpu",
            ).forward_reverse_scaling(2.0, 0.125, 0.5)
            for actual, expected in zip(gathered, reference):
                assert torch.equal(actual, expected)
    finally:
        dist.destroy_process_group()


def test_shard_bounds_cover_uneven_batch():
    assert [shard_bounds(8, rank, 3) for rank in range(3)] == [
        (0, 3),
        (3, 6),
        (6, 8),
    ]


def test_device_controls_default_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _backend(DistributedConfig(2, 2, 2, device="cpu")) == "gloo"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _backend(DistributedConfig(2, 2, 2, device="cuda:1")) == "nccl"


def test_nccl_binds_local_device_before_initializing(monkeypatch):
    calls = []
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: calls.append(("device", device)),
    )
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda backend: calls.append(("group", backend)),
    )

    _process_group(DistributedConfig(2, 2, 2, backend="nccl"))

    assert calls == [
        ("device", torch.device("cuda:3")),
        ("group", "nccl"),
    ]


def test_distributed_tu_matches_batched_solver(tmp_path):
    init_method = "file://{}".format(tmp_path / "process-group")
    mp.spawn(_distributed_worker, args=(2, init_method), nprocs=2, join=True)
