#!/usr/bin/env python3
"""Config-driven dataset preparation, tokenization, adaptation and validation."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.latte_pipeline.data import atomic_json, digest, inspect, prepare, read_config


def setup_evaluation():
    import datetime
    import torch
    import torch.distributed as distributed
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo',
                                       timeout=datetime.timedelta(hours=8))
    torch.set_num_threads(4)
    return rank, world, device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inspect', 'prepare', 'tokenizer-preflight', 'train-tokenizer', 'cache',
                                        'preflight', 'train', 'evaluate', 'submit', 'advance'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--arm', choices=['scratch', 'finetune'], default='finetune')
    parser.add_argument('--account')
    parser.add_argument('--partition', default='standard')
    parser.add_argument('--workflow', type=Path)
    parser.add_argument('--parent')
    parser.add_argument('--after-job')
    args = parser.parse_args()
    if args.stage == 'advance':
        from src.latte_pipeline.scheduler import advance
        advance(args.workflow, args.parent)
        return
    if args.config is None:
        parser.error('--config is required')
    config = read_config(args.config)
    out = Path(config['work_root']).resolve() / config['dataset']
    if args.stage == 'inspect':
        result = inspect(config)
        from src.latte_pipeline.data import parameter_statistics
        print(json.dumps(dict(dataset=config['dataset'], records=len(result['records']),
                              parameters=parameter_statistics(result['records'], config)), indent=2))
        return
    if args.stage == 'submit':
        if not args.account:
            parser.error('--account is required for submission')
        if 'development_architecture' in config or config['scheduler']['gpus'] != 8:
            raise ValueError('Production submission requires eight GPUs and the production architecture')
        from src.latte_pipeline.scheduler import submit
        submit(config, args.config, out, ROOT, args.account, args.partition, args.arm, args.after_job)
        return
    if args.stage == 'prepare':
        if config.get('artifact_source'):
            raise ValueError('Artifact reuse must not regenerate preparation')
        import fcntl
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'prepare.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prepared = prepare(config, out)
            atomic_json(out / 'prepare_complete.json', dict(complete=True, fingerprint=prepared['digest']))
        return
    if config.get('artifact_source') and args.stage in ('tokenizer-preflight', 'train-tokenizer', 'cache'):
        raise ValueError('Artifact reuse only permits forecasting stages')
    if config.get('prerequisite_manifest'):
        from scripts.latte_ablation_sweep import require_core_complete
        require_core_complete(json.loads(Path(config['prerequisite_manifest']).read_text()))
    from src.latte_pipeline import runtime
    rank, world, device = setup_evaluation() if args.stage == 'evaluate' else runtime.setup()
    try:
        if world != config['scheduler']['gpus'] and 'development_architecture' not in config:
            raise ValueError('Use the configured distributed launch; production is eight GPUs')
        if args.stage == 'tokenizer-preflight':
            runtime.tokenizer_preflight(config, out, args.arm, rank, world, device)
        elif args.stage == 'train-tokenizer':
            runtime.train(config, out, args.arm, rank, world, device, tokenizer_training=True)
        else:
            getattr(runtime, args.stage)(config, out, args.arm, rank, world, device)
    finally:
        if runtime.distributed.is_initialized():
            runtime.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
