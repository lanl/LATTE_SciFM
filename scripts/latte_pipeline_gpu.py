#!/usr/bin/env python3
"""Bind only allocation-visible GPUs and verify the distributed device map."""
import os
from pathlib import Path
import subprocess
import sys


def allocated_device(visible, local, world):
    devices = [value.strip() for value in visible.split(',')]
    if len(devices) != world or len(set(devices)) != world or any(not value or value == '-1' for value in devices):
        raise ValueError(f'Expected {world} distinct allocation-visible devices, got {visible!r}')
    if not 0 <= local < world:
        raise ValueError('Invalid local rank')
    return devices[local]


def check():
    import datetime
    import json
    import torch
    import torch.distributed as distributed
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if torch.cuda.device_count() != 1:
        raise ValueError('Worker must see exactly one allocation-local GPU')
    torch.cuda.set_device(0)
    os.environ['MASTER_PORT'] = str(int(os.environ['MASTER_PORT']) + 1)
    os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'False'
    distributed.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    try:
        device = dict(rank=rank, uuid=str(torch.cuda.get_device_properties(0).uuid))
        rows = [None] * world
        distributed.all_gather_object(rows, device)
        if len({row['uuid'] for row in rows}) != world:
            raise ValueError('Ranks share physical GPUs')
        core = torch.nn.Linear(4, 1).cuda()
        model = torch.nn.parallel.DistributedDataParallel(core, device_ids=[0])
        optimizer = torch.optim.SGD(core.parameters(), lr=.01)
        loss = model(torch.full((2, 4), float(rank + 1), device='cuda')).square().mean()
        loss.backward()
        optimizer.step()
        weights = torch.cat([parameter.detach().flatten() for parameter in core.parameters()])
        gathered = [torch.empty_like(weights) for _index in range(world)]
        distributed.all_gather(gathered, weights)
        if not torch.isfinite(weights).all() or any(not torch.equal(weights, value) for value in gathered):
            raise ValueError('Distributed optimizer update diverged')
        if rank == 0:
            print(json.dumps(dict(event='allocation_binding_passed', ranks=rows)), flush=True)
    finally:
        distributed.destroy_process_group()


def main():
    if sys.argv[1:] == ['--check']:
        check()
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = allocated_device(os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                                                         int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE']))
    os.environ['LOCAL_RANK'] = '0'
    os.environ.pop('CUDA_DEVICE_ORDINAL', None)
    subprocess.run([sys.executable, __file__, '--check'], check=True)
    os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name('latte_pipeline.py')), *sys.argv[1:]])


if __name__ == '__main__':
    main()
