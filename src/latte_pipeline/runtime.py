"""Resumable preparation, token caching, training, and generated evaluation."""
import contextlib
import functools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch_ema import ExponentialMovingAverage
import yaml

from .data import artifact_root, atomic_json, digest, file_hash, identity, load_prepared, read_frame, normalize
from .model import ARCHITECTURE, ForecastModel, flatten_codes, migrate_pretrained, token_loss, unflatten_codes
from .tokenizer import make_tokenizer, load_tokenizer, encode_frame, decode_frame, standardized_frame


def setup():
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        local = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    torch.set_num_threads(4)
    return rank, world, device


def barrier():
    if distributed.is_initialized():
        distributed.barrier()


def gather(value, world):
    values = [None] * world
    if world > 1:
        distributed.all_gather_object(values, value)
    else:
        values[0] = value
    return values


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_random(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def code_signature():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / 'src/latte_pipeline').glob('*.py')) + [root / 'scripts/train_encdec_vq_transformer_pdegym6_llama.py',
            root / 'scripts/llama_cached_inference.py'] + list((root / 'third_party/Phaedra/phaedra').glob('*.py'))
    return digest({str(path.relative_to(root)): file_hash(path) for path in sorted(paths)})


def tokenizer_path(config, out, arm):
    return Path(config['pretrained_tokenizer']) if arm == 'finetune' else artifact_root(config, out) / 'tokenizer/best_ema.pt'


def paths(config, out, arm):
    out = Path(out)
    return artifact_root(config, out) / 'cache' / arm, out / arm / config['conditioning']['encoding']


def signature(config, prepared, stage, arm, extra=None):
    if stage.startswith('tokenizer'):
        config = json.loads(json.dumps(config))
        config['conditioning']['encoding'] = 'independent_of_conditioning'
    return dict(version=1, configuration=config, preparation=prepared['digest'], stage=stage, arm=arm,
                code=code_signature(), extra=extra or {})


def require_gate(path, expected):
    gate = json.loads(Path(path).read_text())
    if not gate.get('passed') or gate['signature'] != expected:
        raise ValueError(f'Missing/incompatible preflight: {path}')


def selected_monitor(prepared, config):
    records = [record for record in prepared['records'] if record['split'] == 'valid']
    records = sorted(records, key=lambda record: digest([config['training']['seed'], record['case_id']]))
    maximum = config['frames'] - 1
    horizons = sorted({1, max(1, round(maximum * .25)), max(1, round(maximum * .5)), max(1, round(maximum * .75)), maximum})
    return records[:config['training']['monitor_cases']], horizons


def grouped(values, fields):
    groups = {}
    for index, field in enumerate(fields):
        groups.setdefault(field['group'], []).append(index)
    return np.stack([values[..., indices].mean(-1) for indices in groups.values()], -1).mean(-1)


def physical_metrics(prediction, target):
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if not np.isfinite(difference).all():
        raise ValueError('Nonfinite physical reconstruction or forecast')
    return dict(mse=np.square(difference).mean((-2, -1)), l1=np.abs(difference).mean((-2, -1)),
                mean_bias_abs=np.abs(difference.mean((-2, -1))),
                gradient_mse=(np.square(np.diff(difference, axis=-1)).mean((-2, -1)) +
                              np.square(np.diff(difference, axis=-2)).mean((-2, -1))) / 2)


def global_draws(seed, step, global_batch, count, rank, world):
    if global_batch % world:
        raise ValueError('Global batch must be divisible by world size')
    generator = np.random.default_rng(np.random.SeedSequence([seed, step]))
    return generator.integers(count, size=global_batch)[rank::world]


def tokenizer_preflight(config, out, arm, rank, world, device):
    prepared = load_prepared(config, out)
    configuration = yaml.safe_load(Path(config['tokenizer_config']).read_text())
    if arm == 'finetune':
        model = load_tokenizer(config['pretrained_tokenizer'], device, 98000)
        artifact = identity(config['pretrained_tokenizer'], True)
    else:
        model = make_tokenizer(configuration).to(device)
        artifact = identity(config['tokenizer_config'], True)
    records = [record for record in prepared['records'] if record['split'] == 'train']
    record = records[rank % len(records)]
    fields = config['dynamic_fields'] + config.get('static_fields', [])
    values = list(read_frame(record, config['dynamic_fields'], config['frames'] // 2))
    if config.get('static_fields'):
        values.extend(read_frame(record, config['static_fields']))
    normalized = standardized_frame(np.stack(values), fields, prepared['normalization'])
    if arm == 'scratch':
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for field in normalized:
            tensor = torch.from_numpy(field[None, None]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                reconstruction, commitment, *_ = model(tensor)
                loss = (reconstruction - tensor).abs().mean() + commitment
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError('Nonfinite tokenizer optimizer update')
        del optimizer
    model.eval()
    codes = encode_frame(model, normalized, device)
    reconstructed = decode_frame(model, codes, fields, prepared['normalization'], device)
    physical_metrics(reconstructed, np.stack(values))
    rows = gather(dict(rank=rank, shape=list(codes.shape), finite=True), world)
    if rank == 0:
        atomic_json(Path(out) / f'tokenizer_gate_{arm}.json', dict(passed=True, ranks=rows,
            signature=signature(config, prepared, 'tokenizer-preflight', arm, artifact)))
    barrier()


def cache(config, out, arm, rank, world, device):
    if config.get('artifact_source'):
        raise ValueError('Reused core caches are read-only; do not run cache generation')
    prepared = load_prepared(config, out)
    source = tokenizer_path(config, out, arm)
    if arm == 'scratch':
        completion = json.loads((Path(out) / 'tokenizer/completion.json').read_text())
        if not completion['complete']:
            raise ValueError('Scratch tokenizer is not complete')
    else:
        require_gate(Path(out) / 'tokenizer_gate_finetune.json',
                     signature(config, prepared, 'tokenizer-preflight', arm, identity(source, True)))
    output, _ = paths(config, out, arm)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(version=1, preparation=prepared['digest'], tokenizer=identity(source, True),
                    fields=config['dynamic_fields'], static_fields=config.get('static_fields', []),
                    shape=config['shape'], code=code_signature())
    fingerprint = digest(metadata)
    if rank == 0:
        path = output / 'metadata.json'
        if path.exists() and json.loads(path.read_text()) != metadata:
            raise ValueError('Cache identity changed')
        atomic_json(path, metadata)
    barrier()
    model = load_tokenizer(source, device, 98000 if arm == 'finetune' else None)
    for index, record in enumerate(prepared['records']):
        if index % world != rank:
            continue
        path = output / f'case_{index:05d}.npz'
        if path.exists():
            verify_cache_case(path, record, config, fingerprint)
            continue
        dynamic = []
        for timestep in range(config['frames']):
            frame = read_frame(record, config['dynamic_fields'], timestep)
            dynamic.append(encode_frame(model, standardized_frame(frame, config['dynamic_fields'], prepared['normalization']), device))
        static = np.empty((0, 2, config['shape'][0] // 8, config['shape'][1] // 8), dtype=np.uint16)
        if config.get('static_fields'):
            frame = read_frame(record, config['static_fields'])
            static = encode_frame(model, standardized_frame(frame, config['static_fields'], prepared['normalization']), device)
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, dynamic=np.stack(dynamic), static=static,
                            case_id=record['case_id'], fingerprint=fingerprint)
        verify_cache_case(temporary, record, config, fingerprint)
        temporary.replace(path)
        print(json.dumps(dict(event='cached', rank=rank, case=index)), flush=True)
    barrier()
    if rank == 0:
        for index, record in enumerate(prepared['records']):
            verify_cache_case(output / f'case_{index:05d}.npz', record, config, fingerprint)
        atomic_json(output / 'complete.json', dict(complete=True, fingerprint=fingerprint, cases=len(prepared['records'])))
    barrier()


def verify_cache_case(path, record, config, fingerprint):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['case_id']) != record['case_id'] or str(saved['fingerprint']) != fingerprint:
            raise ValueError('Cache case identity mismatch')
        spatial = (2, config['shape'][0] // 8, config['shape'][1] // 8)
        for key, expected in [('dynamic', (config['frames'], len(config['dynamic_fields'])) + spatial),
                              ('static', (len(config.get('static_fields', [])),) + spatial)]:
            values = saved[key]
            if values.dtype != np.uint16 or values.shape != expected:
                raise ValueError('Incomplete cache geometry/dtype')
            if values.size and (values[..., 0, :, :].max() >= 8640 or values[..., 1, :, :].max() >= 1024):
                raise ValueError('Out-of-range cached token')


class FrameCache:
    def __init__(self, config, prepared, out, arm):
        self.config, self.prepared = config, prepared
        self.root, _ = paths(config, out, arm)
        self.metadata = json.loads((self.root / 'metadata.json').read_text())
        complete = json.loads((self.root / 'complete.json').read_text())
        if complete != dict(complete=True, fingerprint=digest(self.metadata), cases=len(prepared['records'])):
            raise ValueError('Missing/incompatible complete cache')
        expected_code = config.get('artifact_source', {}).get('code', code_signature())
        if self.metadata['preparation'] != prepared['digest'] or self.metadata['code'] != expected_code:
            raise ValueError('Cache preparation/implementation mismatch')
        if (self.metadata['fields'] != config['dynamic_fields'] or
                self.metadata['static_fields'] != config.get('static_fields', []) or
                self.metadata['shape'] != config['shape'] or
                Path(self.metadata['tokenizer']['path']).resolve() != tokenizer_path(config, out, arm).resolve()):
            raise ValueError('Cache fields/geometry/tokenizer mismatch')
        from .data import verify_identity
        verify_identity(self.metadata['tokenizer'])
        self.indices = {record['case_id']: index for index, record in enumerate(prepared['records'])}

    @functools.lru_cache(maxsize=16)
    def arrays(self, index):
        path = self.root / f'case_{index:05d}.npz'
        verify_cache_case(path, self.prepared['records'][index], self.config, digest(self.metadata))
        with np.load(path, allow_pickle=False) as saved:
            return saved['dynamic'], saved['static']

    def example(self, record, horizon, device):
        dynamic, static = self.arrays(self.indices[record['case_id']])
        source = flatten_codes(torch.from_numpy(np.concatenate((dynamic[0], static))[None]).to(device))
        target = flatten_codes(torch.from_numpy(dynamic[horizon:horizon + 1]).to(device))
        parameters = torch.tensor([record['parameters']], dtype=torch.float64, device=device)
        times = self.prepared['coordinates']['time']
        return source, parameters, torch.tensor([times[0]], device=device), torch.tensor([times[horizon]], device=device), target


def create_forecaster(config, prepared, arm, device, initialize=True):
    architecture = config.get('development_architecture', ARCHITECTURE)
    model = ForecastModel(config, prepared['parameter_statistics'], prepared['coordinates']['time'], architecture)
    migration = None
    if arm == 'finetune' and initialize:
        saved = torch.load(config['pretrained_transformer'], map_location='cpu', mmap=True, weights_only=False)
        migration = migrate_pretrained(model, saved)
        del saved
    return model.to(device), migration


def distributed_model(model, world, device):
    if world == 1:
        return model
    return DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                   broadcast_buffers=False)


def forecast_signature(config, prepared, cache_data, arm, world):
    extra = dict(cache=digest(cache_data.metadata), world=world)
    if arm == 'finetune':
        extra['source'] = identity(config['pretrained_transformer'], True)
    return signature(config, prepared, 'forecast', arm, extra)


def preflight(config, out, arm, rank, world, device):
    prepared = load_prepared(config, out)
    cached = FrameCache(config, prepared, out, arm)
    seed_all(config['training']['seed'])
    core, migration = create_forecaster(config, prepared, arm, device)
    model = distributed_model(core, world, device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=1e-5)
    records = [record for record in prepared['records'] if record['split'] == 'train']
    example = cached.example(records[rank % len(records)], config['frames'] - 1, device)
    started = time.monotonic()
    model.train()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logits = model(*example)
        loss = token_loss(logits, example[-1])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(core.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    del logits, optimizer
    if any(not torch.isfinite(parameter).all() for parameter in core.parameters()):
        raise ValueError('Nonfinite forecast optimizer update')
    core.eval()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        generated = core.generate(*example[:-1])
    if generated.shape != example[-1].shape:
        raise ValueError('Incomplete native-grid generation')
    rows = gather(dict(rank=rank, loss=float(loss.detach()), seconds=time.monotonic() - started,
                       peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0), world)
    if rank == 0:
        _, run = paths(config, out, arm)
        atomic_json(run / 'preflight.json', dict(passed=True, signature=forecast_signature(config, prepared, cached, arm, world),
                                                 ranks=rows, migration=migration, update_discarded=True))
    barrier()


def observe(state, score, settings):
    if not math.isfinite(score):
        raise ValueError('Nonfinite validation score')
    improved = state['best'] is None or score < state['best']
    if improved:
        state['best'] = score
    if state['reference'] is None or score < state['reference'] * (1 - settings['relative_improvement']):
        state['reference'], state['stale'] = score, 0
    else:
        state['stale'] += 1
    return improved


@torch.inference_mode()
def tokenizer_score(model, prepared, config, rank, world, device):
    records, horizons = selected_monitor(prepared, config)
    values = []
    model.eval()
    fields = config['dynamic_fields'] + config.get('static_fields', [])
    for index, record in enumerate(records):
        if index % world != rank:
            continue
        for horizon in horizons:
            frame = list(read_frame(record, config['dynamic_fields'], horizon))
            if config.get('static_fields'):
                frame.extend(read_frame(record, config['static_fields']))
            normalized = standardized_frame(np.stack(frame), fields, prepared['normalization'])
            for field in normalized:
                tensor = torch.from_numpy(field[None, None]).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    reconstructed = model(tensor)[0]
                values.append(float((reconstructed.float() - tensor).square().mean()))
    return float(np.mean([value for shard in gather(values, world) for value in shard]))


@torch.inference_mode()
def forecast_score(model, tokenizer, cached, prepared, config, rank, world, device):
    records, horizons = selected_monitor(prepared, config)
    values = []
    fields = config['dynamic_fields']
    scales = np.asarray([prepared['normalization'][field['name']]['physical_scale'] for field in fields])
    model.eval()
    for index, record in enumerate(records):
        if index % world != rank:
            continue
        for horizon in horizons:
            example = cached.example(record, horizon, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                sequence = model.generate(*example[:-1])
            codes = unflatten_codes(sequence, len(fields), model.Hq, model.Wq)[0].cpu().numpy()
            prediction = decode_frame(tokenizer, codes, fields, prepared['normalization'], device)
            target = read_frame(record, fields, horizon)
            mse = physical_metrics(prediction, target)['mse']
            values.append(float(grouped(mse / scales ** 2, fields)))
    return float(np.mean([value for shard in gather(values, world) for value in shard]))


def train(config, out, arm, rank, world, device, tokenizer_training=False):
    prepared = load_prepared(config, out)
    settings = config['training']
    seed_all(settings['seed'])
    tokenizer = None
    if tokenizer_training:
        if arm != 'scratch':
            raise ValueError('Pretrained tokenizers are always frozen')
        configuration = yaml.safe_load(Path(config['tokenizer_config']).read_text())
        require_gate(Path(out) / 'tokenizer_gate_scratch.json', signature(config, prepared, 'tokenizer-preflight', arm,
                                                                      identity(config['tokenizer_config'], True)))
        run = Path(out) / 'tokenizer'
        protocol = signature(config, prepared, 'tokenizer', arm, dict(world=world, model=configuration))
        core = make_tokenizer(configuration).to(device)
        from phaedra.edemamix import AdEMAMix
        optimizer = AdEMAMix(core.parameters(), lr=settings['tokenizer_lr'], betas=(.5, .9, .99), alpha=2.,
                            beta3_warmup=250, alpha_warmup=250, weight_decay=.01)
        ema = ExponentialMovingAverage(core.parameters(), decay=.999)
        cached = None
    else:
        cached = FrameCache(config, prepared, out, arm)
        _, run = paths(config, out, arm)
        protocol = forecast_signature(config, prepared, cached, arm, world)
        require_gate(run / 'preflight.json', protocol)
        core, _ = create_forecaster(config, prepared, arm, device, initialize=not (run / 'last.pt').exists())
        optimizer = torch.optim.AdamW(core.parameters(), lr=settings[arm + '_lr'], weight_decay=.01)
        tokenizer = load_tokenizer(tokenizer_path(config, out, arm), device, 98000 if arm == 'finetune' else None)
        ema = None
    run.mkdir(parents=True, exist_ok=True)
    completion = run / 'completion.json'
    if completion.exists():
        existing = json.loads(completion.read_text())
        if existing['signature'] != protocol or not existing['complete']:
            raise ValueError('Incompatible completion record')
        return
    start, state = 0, dict(best=None, reference=None, stale=0, best_step=None)
    resume_rng = None
    if (run / 'last.pt').exists():
        saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
        if saved['signature'] != protocol or len(saved['rng']) != world:
            raise ValueError('Resume configuration/code/cache/world mismatch')
        core.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        if ema:
            ema.load_state_dict(saved['ema'])
        start, state, resume_rng = saved['step'], saved['plateau'], saved['rng'][rank]
        del saved
    model = distributed_model(core, world, device)
    if resume_rng:
        restore_random(resume_rng)
    else:
        seed_all(settings['seed'] + rank)
    if rank == 0:
        atomic_json(run / 'protocol.json', protocol)
    records = [record for record in prepared['records'] if record['split'] == 'train']
    fields = config['dynamic_fields'] + config.get('static_fields', [])
    field_examples = len(fields) * config['frames']
    examples_per_record = field_examples if tokenizer_training else config['frames'] - 1
    if settings['global_batch'] % world:
        raise ValueError('Global batch must divide world size')
    accumulator = settings['global_batch'] // world
    for step in range(start + 1, settings['max_steps'] + 1):
        if state['stale'] >= settings['patience']:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if not tokenizer_training:
            rate = settings[arm + '_lr'] * min(1., step / max(1, settings['warmup']))
            for group in optimizer.param_groups:
                group['lr'] = rate
        losses = []
        draws = global_draws(settings['seed'], step, settings['global_batch'], len(records) * examples_per_record, rank, world)
        for micro, draw in enumerate(draws):
            case, selection = divmod(int(draw), examples_per_record)
            synchronize = micro == accumulator - 1 or world == 1
            context = contextlib.nullcontext() if synchronize else model.no_sync()
            with context:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    if tokenizer_training:
                        field_index, horizon = divmod(selection, config['frames'])
                        field = fields[field_index]
                        if field_index >= len(config['dynamic_fields']):
                            horizon = None
                        raw = read_frame(records[case], [field], horizon)[0]
                        tensor = torch.from_numpy(normalize(raw, prepared['normalization'][field['name']])[None, None]).to(device)
                        reconstructed, commitment, *_ = model(tensor)
                        loss = (reconstructed - tensor).abs().mean() + commitment
                    else:
                        example = cached.example(records[case], selection + 1, device)
                        loss = token_loss(model(*example), example[-1])
                (loss / accumulator).backward()
                losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if ema:
            ema.update()
        improved = False
        if step % settings['eval_every'] == 0 or step == settings['max_steps']:
            if ema:
                with ema.average_parameters():
                    score = tokenizer_score(core, prepared, config, rank, world, device)
            else:
                score = forecast_score(core, tokenizer, cached, prepared, config, rank, world, device)
            improved = observe(state, score, settings)
            if improved:
                state['best_step'] = step
            if rank == 0:
                atomic_json(run / f'validation_{step:06d}.json', dict(step=step, score=score, plateau=state,
                            metric='normalized reconstruction MSE' if ema else 'generated physical grouped MSE / train variance'))
        if step % settings['save_every'] == 0 or improved or step == settings['max_steps'] or state['stale'] >= settings['patience']:
            states = gather(random_state(), world)
            if rank == 0:
                saved = dict(step=step, model=core.state_dict(), optimizer=optimizer.state_dict(), plateau=state,
                             rng=states, signature=protocol, sampler=dict(seed=settings['seed'], next_step=step + 1))
                if ema:
                    saved.update(ema=ema.state_dict(), config=configuration, mse=state['best'])
                if improved:
                    atomic_checkpoint(run / ('best_ema.pt' if ema else 'best.pt'), saved)
                atomic_checkpoint(run / 'last.pt', saved)
            barrier()
        if rank == 0 and (step % 25 == 0 or step == start + 1):
            print(json.dumps(dict(event='train', tokenizer=tokenizer_training, step=step, loss=float(np.mean(losses)),
                                  best=state['best'], stale=state['stale'])), flush=True)
    if state['best_step'] is None:
        raise ValueError('Training ended without a selected checkpoint')
    if rank == 0:
        atomic_json(completion, dict(complete=True, signature=protocol, plateau=state,
                                    reason='plateau' if state['stale'] >= settings['patience'] else 'step_cap'))
    barrier()


def evaluate(config, out, arm, rank, world, device):
    prepared = load_prepared(config, out)
    cached = FrameCache(config, prepared, out, arm)
    _, run = paths(config, out, arm)
    completion = json.loads((run / 'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Training is not complete')
    core, _ = create_forecaster(config, prepared, arm, device, initialize=False)
    saved = torch.load(run / 'best.pt', map_location='cpu', mmap=True, weights_only=False)
    if saved['signature'] != forecast_signature(config, prepared, cached, arm, world):
        raise ValueError('Evaluation checkpoint signature changed')
    core.load_state_dict(saved['model'], strict=True)
    core.eval().requires_grad_(False)
    selected_step = saved['step']
    del saved
    tokenizer = load_tokenizer(tokenizer_path(config, out, arm), device, 98000 if arm == 'finetune' else None)
    output = run / 'validation'
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(dict(checkpoint=identity(run / 'best.pt', True), preparation=prepared['digest']))
    records = [record for record in prepared['records'] if record['split'] == 'valid']
    fields = config['dynamic_fields']
    for index, record in enumerate(records):
        if index % world != rank:
            continue
        destination = output / f'case_{index:04d}.npz'
        if destination.exists():
            verify_evaluation_case(destination, record, config, fingerprint)
            continue
        rows, images = {}, {}
        histograms = [np.zeros(8640, dtype=np.int64), np.zeros(1024, dtype=np.int64)]
        for horizon in range(1, config['frames']):
            example = cached.example(record, horizon, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                sequence = core.generate(*example[:-1])
            codes = unflatten_codes(sequence, len(fields), core.Hq, core.Wq)[0].cpu().numpy()
            prediction = decode_frame(tokenizer, codes, fields, prepared['normalization'], device)
            target = read_frame(record, fields, horizon)
            native_codes = cached.arrays(cached.indices[record['case_id']])[0][horizon]
            reconstruction = decode_frame(tokenizer, native_codes, fields, prepared['normalization'], device)
            for family in range(2):
                histograms[family] += np.bincount(native_codes[:, family].reshape(-1), minlength=len(histograms[family]))
            for name, values in [('forecast', prediction), ('reconstruction', reconstruction)]:
                for metric, value in physical_metrics(values, target).items():
                    rows.setdefault(name + '_' + metric, []).append(value)
            if index == 0 and horizon in selected_monitor(prepared, config)[1]:
                images[f'target_{horizon}'] = target
                images[f'forecast_{horizon}'] = prediction
                images[f'reconstruction_{horizon}'] = reconstruction
        temporary = destination.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **{key: np.stack(value) for key, value in rows.items()},
                            morphology_counts=histograms[0], amplitude_counts=histograms[1],
                            case_id=record['case_id'], fingerprint=fingerprint)
        verify_evaluation_case(temporary, record, config, fingerprint)
        temporary.replace(destination)
        if index == 0:
            np.savez_compressed(output / 'sample.npz', **images)
        print(json.dumps(dict(event='evaluated', case=index, arm=arm)), flush=True)
    barrier()
    if rank == 0:
        arrays = {}
        for index, record in enumerate(records):
            path = output / f'case_{index:04d}.npz'
            verify_evaluation_case(path, record, config, fingerprint)
            with np.load(path, allow_pickle=False) as saved:
                for key in saved.files:
                    if key not in ('fingerprint', 'case_id'):
                        arrays.setdefault(key, []).append(saved[key])
        arrays = {key: np.stack(value) for key, value in arrays.items()}
        np.savez_compressed(output / 'metrics.npz', **arrays)
        scales = np.asarray([prepared['normalization'][field['name']]['physical_scale'] for field in fields])
        summary = dict(complete=True, dataset=config['dataset'], arm=arm, encoding=config['conditioning']['encoding'],
                       selected_step=selected_step, cases=len(records), horizons=config['frames'] - 1,
                       times=prepared['coordinates']['time'][1:], fields=fields, fingerprint=fingerprint, test_used=False)
        for name in ('forecast', 'reconstruction'):
            mse = arrays[name + '_mse']
            summary[name] = dict(physical_grouped_mse=float(grouped(mse, fields).mean()),
                normalized_grouped_mse=float(grouped(mse / scales ** 2, fields).mean()),
                curve=grouped(mse, fields).mean(0).tolist(), per_field_mse=mse.mean((0, 1)).tolist(),
                per_field_rmse=np.sqrt(mse.mean((0, 1))).tolist(),
                per_field_l1=arrays[name + '_l1'].mean((0, 1)).tolist(),
                per_field_gradient_mse=arrays[name + '_gradient_mse'].mean((0, 1)).tolist(),
                per_field_mean_bias_abs=arrays[name + '_mean_bias_abs'].mean((0, 1)).tolist())
        summary['token_usage'] = {}
        for family in ('morphology', 'amplitude'):
            counts = arrays[family + '_counts'].sum(0)
            probability = counts[counts > 0] / counts.sum()
            summary['token_usage'][family] = dict(used=int((counts > 0).sum()), perplexity=float(np.exp(-(probability * np.log(probability)).sum())))
        atomic_json(output / 'summary.json', summary)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 4))
        for name in ('forecast', 'reconstruction'):
            axis.plot(summary['times'], summary[name]['curve'], label=name)
        axis.set(xlabel='Physical time', ylabel='Physical grouped MSE', title=f"{config['dataset']} / {arm}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / 'validation_curve.png', dpi=180)
        plt.close(figure)
    barrier()


def verify_evaluation_case(path, record, config, fingerprint):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['fingerprint']) != fingerprint or str(saved['case_id']) != record['case_id']:
            raise ValueError('Evaluation identity changed')
        for method in ('forecast', 'reconstruction'):
            for metric in ('mse', 'l1', 'gradient_mse', 'mean_bias_abs'):
                values = saved[method + '_' + metric]
                if values.shape != (config['frames'] - 1, len(config['dynamic_fields'])) or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Incomplete/nonfinite evaluation coverage')
