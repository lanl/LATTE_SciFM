#!/usr/bin/env python3
"""Register isolated conditioning ablations behind a complete-core-sweep barrier."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.latte_pipeline.data import atomic_json, digest, read_config
from src.latte_pipeline.scheduler import slurm_environment, submit

DATASETS = ('acoustics', 'cloverleaf', 'gray_scott', 'kolmogorov')
ARMS = ('scratch', 'finetune')
ENCODINGS = ('none', 'continuous')


def core_status(manifest):
    dependencies = []
    if len(manifest['core']) != 8:
        raise ValueError('Require all eight core workflows')
    for entry in manifest['core']:
        path = Path(entry['workflow'])
        workflow = json.loads(path.read_text())
        if workflow['configuration_digest'] != entry['configuration_digest']:
            raise ValueError(f'Core protocol changed: {path}')
        if workflow.get('stopped'):
            raise RuntimeError(f'Core workflow stopped: {path}: {workflow.get("reason")}')
        if workflow.get('complete'):
            run = Path(workflow['out']) / workflow['arm'] / 'binned'
            completion = json.loads((run / 'completion.json').read_text())
            summary = json.loads((run / 'validation/summary.json').read_text())
            if not completion.get('complete') or not summary.get('complete') or summary.get('cases') != entry['validation_cases']:
                raise ValueError(f'Core training/full validation incomplete: {path}')
        else:
            current = workflow['history'][-1]
            if current['job'] != workflow.get('active_job') or not str(current.get('watcher', '')).isdigit():
                raise RuntimeError(f'Core continuation not fully registered: {path}')
            dependencies.append(str(current['watcher']))
    return sorted(set(dependencies))


def require_core_complete(manifest):
    if core_status(manifest):
        raise RuntimeError('The entire core sweep, including full validation, must finish first')


def prepare(core_root, output, account, partition):
    core_root, output = core_root.resolve(), output.resolve()
    if output == core_root or core_root in output.parents or output in core_root.parents:
        raise ValueError('Core and ablation roots must be separate directory trees')
    path = output / 'sweep.json'
    if path.exists():
        raise FileExistsError(f'Existing sweep: {path}; use submit/monitor instead')
    manifest = dict(version=1, repo=str(ROOT), revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                    text=True).strip(), python=sys.executable, account=account, partition=partition,
                    core=[], experiments=[], registered=False, released=False)
    configurations = {}
    for dataset in DATASETS:
        codes = set()
        for arm in ARMS:
            workflow_path = core_root / dataset / f'workflow_{arm}.json'
            workflow = json.loads(workflow_path.read_text())
            config = read_config(workflow['config'])
            if (workflow['dataset'] != dataset or workflow['arm'] != arm or config['conditioning']['encoding'] != 'binned'
                    or digest(config) != workflow['configuration_digest']):
                raise ValueError('Core workflow/configuration mismatch')
            manifest['core'].append(dict(workflow=str(workflow_path), configuration_digest=digest(config),
                                         validation_cases=config['splits']['valid']))
            codes.add(json.loads((core_root / dataset / 'cache' / arm / 'metadata.json').read_text())['code'])
            if dataset in configurations and configurations[dataset] != config:
                raise ValueError('Core arms must share their configuration')
            configurations[dataset] = config
        if len(codes) != 1:
            raise ValueError('Core caches have inconsistent producer signatures')
        for encoding in ENCODINGS:
            config = dict(configurations[dataset])
            config['conditioning'] = dict(config['conditioning'], encoding=encoding)
            config['work_root'] = str(output / encoding)
            config['artifact_source'] = dict(root=str(core_root / dataset), code=next(iter(codes)))
            config['prerequisite_manifest'] = str(path)
            location = output / 'configs' / f'{dataset}_{encoding}.yaml'
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_text(yaml.safe_dump(config, sort_keys=False))
            for arm in ARMS:
                manifest['experiments'].append(dict(config=str(location), arm=arm, configuration_digest=digest(config),
                    workflow=str(output / encoding / dataset / f'workflow_{arm}.json')))
    core_status(manifest)
    atomic_json(path, manifest)
    print(str(path))


def verify_manifest(manifest):
    if len(manifest['experiments']) != 16:
        raise ValueError('Require sixteen ablation experiments')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=manifest['repo'], text=True).strip()
    if revision != manifest['revision'] or str(ROOT) != manifest['repo']:
        raise ValueError('Ablation checkout changed')
    for entry in manifest['experiments']:
        if digest(read_config(entry['config'])) != entry['configuration_digest']:
            raise ValueError('Ablation configuration changed')


def cpu_job(manifest, path, action, dependencies=(), held=False):
    command = [manifest['python'], str(ROOT / 'scripts/latte_ablation_sweep.py'), action, '--manifest', str(path)]
    logs = path.parent / 'gate_logs'
    logs.mkdir(parents=True, exist_ok=True)
    arguments = ['sbatch', '--parsable', '--partition', manifest['partition'], '--account', manifest['account'],
                 '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G', '--gres=none', '--time=00:20:00',
                 '--no-requeue', '--export=ALL', '--chdir', manifest['repo'], '--job-name=latte-ablation-' + action,
                 '--output', str(logs / '%x-%j.out'), '--error', str(logs / '%x-%j.err')]
    if held:
        arguments += ['--hold']
    if dependencies:
        arguments += ['--dependency=afterany:' + ':'.join(dependencies), '--kill-on-invalid-dep=yes']
    result = subprocess.check_output(arguments + ['--wrap', shlex.join(command)], env=slurm_environment(), text=True).strip().split(';')[0]
    if not result.isdigit():
        raise ValueError('Invalid CPU gate job ID')
    return result


def advance_gate(path, manifest):
    if manifest.get('released'):
        return
    dependencies = core_status(manifest)
    if dependencies:
        if set(dependencies) & set(manifest.get('waited_for', [])):
            raise RuntimeError('A completed core watcher did not advance its workflow; inspect instead of looping')
        manifest['monitor_job'] = cpu_job(manifest, path, 'monitor', dependencies)
        manifest['waited_for'] = dependencies
        atomic_json(path, manifest)
        print(json.dumps(dict(monitor=manifest['monitor_job'], waits_for=dependencies)))
    else:
        state = subprocess.check_output(['scontrol', 'show', 'job', manifest['barrier_job'], '-o'], text=True)
        if 'JobState=PENDING' in state and 'Reason=JobHeldUser' in state:
            subprocess.run(['scontrol', 'release', manifest['barrier_job']], check=True)
        elif not any('JobState=' + value in state for value in ('RUNNING', 'COMPLETED', 'PENDING')):
            raise RuntimeError('Barrier job is not runnable; inspect it before proceeding')
        manifest['released'] = True
        atomic_json(path, manifest)
        print(json.dumps(dict(released=manifest['barrier_job'])))


def register(path, manifest):
    if manifest.get('registered'):
        print(json.dumps(dict(already_registered=True, barrier=manifest['barrier_job'], monitor=manifest.get('monitor_job'))))
        return
    core_status(manifest)
    if 'barrier_job' not in manifest:
        manifest['barrier_job'] = cpu_job(manifest, path, 'check', held=True)
        atomic_json(path, manifest)
    for entry in manifest['experiments']:
        config = read_config(entry['config'])
        workflow_path = Path(entry['workflow'])
        if workflow_path.exists():
            workflow = json.loads(workflow_path.read_text())
            if (workflow.get('after_job') != manifest['barrier_job'] or
                    workflow['configuration_digest'] != entry['configuration_digest'] or
                    not workflow['history'][-1].get('watcher')):
                raise RuntimeError('Partial/incompatible registration; inspect workflow before retrying')
        else:
            submit(config, entry['config'], workflow_path.parent, ROOT, manifest['account'], manifest['partition'],
                   entry['arm'], manifest['barrier_job'])
    manifest['registered'] = True
    atomic_json(path, manifest)
    advance_gate(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'monitor', 'check'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--core-root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--account')
    parser.add_argument('--partition', default='standard')
    args = parser.parse_args()
    if args.action == 'prepare':
        if not all((args.core_root, args.out, args.account)):
            parser.error('prepare requires --core-root, --out, and --account')
        prepare(args.core_root, args.out, args.account, args.partition)
        return
    if args.manifest is None:
        parser.error('--manifest is required')
    path = args.manifest.resolve()
    with open(path.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(path.read_text())
        verify_manifest(manifest)
        if args.action == 'check':
            if not manifest['registered']:
                raise RuntimeError('Ablations are not fully registered')
            require_core_complete(manifest)
            print('All eight core workflows and full validations completed; ablation barrier passed.')
        elif args.action == 'submit':
            register(path, manifest)
        else:
            advance_gate(path, manifest)


if __name__ == '__main__':
    main()
