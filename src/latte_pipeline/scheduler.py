"""Durable CPU continuations: advance completed stages, retry timeouts only."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .data import atomic_json, digest


def stages(arm):
    result = ['prepare', 'tokenizer-preflight']
    if arm == 'scratch':
        result.append('train-tokenizer')
    return result + ['cache', 'preflight', 'train', 'evaluate']


def terminal_action(state):
    if state == 'COMPLETED':
        return 'advance'
    if state == 'TIMEOUT':
        return 'retry'
    return 'stop'


def slurm_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(('SLURM_MEM_', 'SLURM_TRES_', 'SLURM_GPUS', 'SLURM_CPUS_'))
            and key not in ('SLURM_NTASKS', 'SLURM_NTASKS_PER_NODE', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDINAL')}


def launch(workflow_path, workflow):
    root = Path(workflow['repo'])
    stage = workflow['stages'][workflow['index']]
    output = Path(workflow['out'])
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    common = ['sbatch', '--parsable', '--partition', workflow['partition'], '--account', workflow['account'],
              '--nodes=1', '--ntasks=1', '--no-requeue', '--export=ALL', '--chdir', str(root)]
    arguments = [stage, '--config', workflow['config'], '--arm', workflow['arm']]
    launcher = root / 'sbatch/sbatch_latte_pipeline_stage.sh'
    resources = ['--cpus-per-task=8', '--mem=32G', '--gres=none'] if stage == 'prepare' else [
        '--cpus-per-task=32', '--mem-per-gpu=80G', '--gres=gpu:nvidia_h100_80gb_hbm3:8',
        '--gpus-per-task=nvidia_h100_80gb_hbm3:8']
    environment = slurm_environment()
    environment['LATTE_PYTHON'] = workflow['python']
    dependency = ['--dependency=afterok:' + workflow['after_job'], '--kill-on-invalid-dep=yes'] if workflow.get('after_job') and not workflow.get('history') else []
    label = '-' + workflow['encoding'] if workflow.get('reuse_artifacts') else ''
    job = subprocess.check_output(common + resources + dependency + ['--time=08:00:00', '--job-name', f"latte-{workflow['dataset']}-{workflow['arm']}{label}-{stage}",
        '--output', str(logs / '%x-%j.out'), '--error', str(logs / '%x-%j.err'), str(launcher), *arguments],
        text=True, env=environment).strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Invalid sbatch job response')
    workflow['active_job'] = job
    workflow.setdefault('history', []).append(dict(job=job, stage=stage, submitted=time.time()))
    atomic_json(workflow_path, workflow)
    command = shlex.join([workflow['python'], str(root / 'scripts/latte_pipeline.py'), 'advance', '--workflow',
                          str(workflow_path), '--parent', job])
    watcher = subprocess.check_output(common + ['--cpus-per-task=1', '--mem=4G', '--gres=none', '--time=00:20:00',
        '--job-name=latte-stage-watch', '--dependency=afterany:' + job, '--kill-on-invalid-dep=yes',
        '--output', str(logs / 'watch-%j.out'), '--error', str(logs / 'watch-%j.err'), '--wrap', command],
        text=True, env=environment).strip().split(';')[0]
    workflow['history'][-1]['watcher'] = watcher
    atomic_json(workflow_path, workflow)
    print(json.dumps(dict(job=job, watcher=watcher, stage=stage, arm=workflow['arm'])), flush=True)


def submit(config, config_path, out, repo, account, partition, arm, after_job=None):
    import fcntl
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'workflow_{arm}.json'
    with open(out / f'workflow_{arm}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            raise FileExistsError(f'Workflow already exists: {path}; inspect rather than duplicate jobs')
        workflow = dict(repo=str(repo), config=str(Path(config_path).resolve()), out=str(out), dataset=config['dataset'],
                        python=sys.executable, account=account, partition=partition, arm=arm,
                        stages=['preflight', 'train', 'evaluate'] if config.get('artifact_source') else stages(arm), index=0,
                        encoding=config['conditioning']['encoding'], reuse_artifacts=bool(config.get('artifact_source')), after_job=after_job,
                        configuration_digest=digest(config), revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                        cwd=repo, text=True).strip(), environment={key: value for key, value in os.environ.items()
                        if key.startswith('LATTE_') and key not in ('LATTE_PYTHON',)})
        atomic_json(path, workflow)
        launch(path, workflow)


def advance(path, parent):
    import fcntl
    from .data import read_config
    path = Path(path).resolve()
    with open(path.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        workflow = json.loads(path.read_text())
        if workflow.get('active_job') != parent:
            raise ValueError('Stale/duplicate continuation')
        os.environ.update(workflow['environment'])
        if digest(read_config(workflow['config'])) != workflow['configuration_digest']:
            raise ValueError('Configuration changed after submission')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workflow['repo'], text=True).strip()
        if revision != workflow['revision']:
            raise ValueError('Pinned checkout revision changed')
        state = ''
        for _attempt in range(60):
            output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', parent, '--format=JobIDRaw,State'], text=True)
            states = [row.split('|')[1].split()[0] for row in output.splitlines() if row.split('|')[0] == parent]
            state = states[0] if states else ''
            if state in ('COMPLETED', 'TIMEOUT', 'FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'):
                break
            time.sleep(5)
        action = terminal_action(state)
        workflow['history'][-1]['terminal_state'] = state
        if action == 'advance':
            workflow['index'] += 1
            if workflow['index'] == len(workflow['stages']):
                workflow.update(complete=True, active_job=None)
                atomic_json(path, workflow)
                return
        elif action == 'stop':
            workflow.update(stopped=True, reason=state or 'unknown_accounting_state')
            atomic_json(path, workflow)
            return
        atomic_json(path, workflow)
        launch(path, workflow)
