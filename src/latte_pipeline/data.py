"""Audited, train-only preparation of named Well-style HDF5 fields."""
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def identity(path, hashed=False):
    path = Path(path).resolve()
    stats = path.stat()
    result = dict(path=str(path), size=stats.st_size, mtime_ns=stats.st_mtime_ns)
    if hashed:
        result['sha256'] = file_hash(path)
    return result


def verify_identity(value):
    if identity(value['path'], 'sha256' in value) != value:
        raise ValueError(f"Artifact changed: {value['path']}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def read_config(path):
    path = Path(path).resolve()
    def load_yaml(location):
        values = yaml.safe_load(os.path.expandvars(location.read_text()))
        parent = values.pop('defaults', None)
        if parent:
            base = load_yaml(location.parent / parent)
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    base[key] = dict(base[key], **value)
                else:
                    base[key] = value
            return base
        return values
    config = load_yaml(path)
    if '${' in json.dumps(config):
        raise ValueError('Unresolved environment variable in configuration')
    if config.get('version') != 1 or set(config['splits']) != {'train', 'valid'}:
        raise ValueError('Require version1 and train/valid only; test access is not enabled')
    fields = config['dynamic_fields'] + config.get('static_fields', [])
    if not fields or len({field['name'] for field in fields}) != len(fields):
        raise ValueError('Require distinct named fields')
    for field in fields:
        if not field['path'].startswith(('t0_fields/', 't1_fields/', 't2_fields/')):
            raise ValueError('Only explicitly named physical fields are supported')
    if config['conditioning']['encoding'] not in ('binned', 'continuous', 'none'):
        raise ValueError('Unknown conditioning encoding')
    if any(not parameter['path'].startswith('scalars/') for parameter in config['parameters']):
        raise ValueError('Controls must come from scalars, not provenance/diagnostics')
    if len({parameter['name'] for parameter in config['parameters']}) != len(config['parameters']):
        raise ValueError('Duplicate parameter names')
    if any(size % 8 for size in config['shape']):
        raise ValueError('Native dimensions must be multiples of df8')
    return config


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def field_array(handle, field, sample, times=None):
    dataset = handle[field['path']]
    varying = bool(dataset.attrs['time_varying'])
    if not bool(dataset.attrs.get('sample_varying', False)):
        raise ValueError('Version1 requires sample-varying physical fields')
    if varying != (times is not None):
        raise ValueError(f"Incorrect dynamic/static role: {field['name']}")
    selection = (sample, times) if varying else (sample,)
    if 'component' in field:
        selection += (slice(None), slice(None), int(field['component']))
    values = np.asarray(dataset[selection])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite physical field')
    return values


def inspect(config):
    root = Path(config['data_root']) / config['directory']
    records, artifacts, all_ids, coordinate_reference = [], {}, set(), None
    for split, expected in config['splits'].items():
        files = sorted((root / 'data' / split).glob('*.h5'))
        if not files:
            raise ValueError(f'No readable shards for {split}')
        count = 0
        for path in files:
            artifacts[str(path)] = identity(path)
            with h5py.File(path, 'r') as handle:
                identifiers = handle['provenance/case_id'][()]
                coordinates = {axis: np.asarray(handle[f'dimensions/{axis}'][()]) for axis in ('x', 'y', 'time')}
                if [len(coordinates[axis]) for axis in ('x', 'y')] != config['shape']:
                    raise ValueError('Unexpected native grid; resizing is disabled')
                if len(coordinates['time']) != config['frames'] or not (np.diff(coordinates['time']) > 0).all():
                    raise ValueError('Invalid physical time grid')
                current = {axis: values.tolist() for axis, values in coordinates.items()}
                if coordinate_reference is None:
                    coordinate_reference = current
                elif current != coordinate_reference:
                    raise ValueError('Version1 requires a common coordinate/time grid across shards')
                for role, fields in [('dynamic', config['dynamic_fields']), ('static', config.get('static_fields', []))]:
                    for field in fields:
                        values = handle[field['path']]
                        expected_shape = (len(identifiers),) + ((config['frames'],) if role == 'dynamic' else ()) + tuple(config['shape'])
                        if bool(values.attrs.get('time_varying', False)) != (role == 'dynamic'):
                            raise ValueError('Field role does not match HDF5 attributes')
                        if not bool(values.attrs.get('sample_varying', False)):
                            raise ValueError('Unsupported shared field layout')
                        if 'component' in field:
                            if values.shape[:-1] != expected_shape or not 0 <= field['component'] < values.shape[-1]:
                                raise ValueError('Invalid vector component/layout')
                        elif values.shape != expected_shape:
                            raise ValueError('Invalid scalar field layout')
                for sample, case_id in enumerate(identifiers):
                    case_id = text(case_id)
                    if case_id in all_ids:
                        raise ValueError('Duplicate case ID or cross-split overlap')
                    all_ids.add(case_id)
                    parameters = []
                    for parameter in config['parameters']:
                        scalar = handle[parameter['path']]
                        if bool(scalar.attrs.get('time_varying', False)):
                            raise ValueError('Time-varying controls require a schedule adapter')
                        value = scalar[sample] if bool(scalar.attrs.get('sample_varying', False)) else scalar[()]
                        value = float(np.asarray(value).reshape(()))
                        if not np.isfinite(value):
                            raise ValueError('Required physical parameter is missing/nonfinite')
                        parameters.append(value)
                    records.append(dict(file=str(path), sample=sample, case_id=case_id, split=split, parameters=parameters))
                    count += 1
        if count != expected:
            raise ValueError(f'{split}: expected {expected} trajectories, found {count}')
    return dict(config=config, records=records, artifacts=artifacts, coordinates=coordinate_reference)


class Moments:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0., 0.
        self.minimum, self.maximum = float('inf'), -float('inf')

    def update(self, array):
        array = np.asarray(array, dtype=np.float64)
        count, mean = array.size, float(array.mean())
        delta, total = mean - self.mean, self.count + count
        self.m2 += float(np.square(array - mean).sum()) + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def result(self):
        span = self.maximum - self.minimum
        deviation = (self.m2 / self.count) ** .5
        return dict(count=self.count, minimum=self.minimum, maximum=self.maximum, span=span or 1.,
                    physical_mean=self.mean, physical_scale=max(deviation, 1e-6),
                    mean=2 * (self.mean - self.minimum) / (span or 1.) - 1,
                    scale=max(2 * deviation / (span or 1.), 1e-6), constant=span == 0)


def normalize(values, stats):
    unit = np.clip((np.asarray(values, dtype=np.float32) - np.float32(stats['minimum'])) / np.float32(stats['span']), 0, 1)
    return ((2 * unit - 1 - np.float32(stats['mean'])) / np.float32(stats['scale'])).astype(np.float32)


def physical(values, stats, clip=True):
    unit = np.asarray(values, dtype=np.float32) * np.float32(stats['scale']) + np.float32(stats['mean'])
    if clip:
        unit = np.clip(unit, -1, 1)
    return ((unit + 1) * .5 * np.float32(stats['span']) + np.float32(stats['minimum'])).astype(np.float32)


def bin_values(values, bounds, bins):
    values = np.asarray(values, dtype=np.float64)
    minimum, maximum = bounds['minimum'], bounds['maximum']
    output = np.zeros(values.shape, dtype=np.int64)
    finite = np.isfinite(values)
    inside = finite & (values >= minimum) & (values <= maximum)
    if maximum > minimum:
        output[inside] = 1 + np.rint((values[inside] - minimum) / (maximum - minimum) * (bins - 1)).astype(np.int64)
    else:
        output[inside] = 1
    output[finite & (values < minimum)] = bins + 1
    output[finite & (values > maximum)] = bins + 2
    return output


def parameter_statistics(records, config):
    values = np.asarray([record['parameters'] for record in records if record['split'] == 'train'], dtype=np.float64)
    results = []
    for index, parameter in enumerate(config['parameters']):
        column = values[:, index]
        result = dict(name=parameter['name'], unit=parameter['unit'], minimum=float(column.min()), maximum=float(column.max()),
                      mean=float(column.mean()), scale=max(float(column.std()), 1e-12), count=len(column),
                      distinct_values=len(np.unique(column)), quantiles=np.quantile(column, [0, .25, .5, .75, 1]).tolist())
        tokens = bin_values(column, result, config['conditioning']['bins'])
        result['distinct_bins'] = len(np.unique(tokens))
        result['merged_distinct_values'] = result['distinct_values'] - result['distinct_bins']
        result['bin_counts'] = np.bincount(tokens, minlength=config['conditioning']['bins'] + 3).tolist()
        results.append(result)
    return results


def prepare(config, out):
    inspected = inspect(config)
    signature = digest(inspected)
    path = Path(out) / 'prepared.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if data_configuration(existing['config']) != data_configuration(config) or existing['artifacts'] != inspected['artifacts']:
            raise ValueError('Preparation inputs changed; use a new output directory')
        return existing
    fields = config['dynamic_fields'] + config.get('static_fields', [])
    moments = {field['name']: Moments() for field in fields}
    for record in inspected['records']:
        if record['split'] != 'train':
            continue
        with h5py.File(record['file'], 'r') as handle:
            for field in config['dynamic_fields']:
                for start in range(0, config['frames'], 16):
                    moments[field['name']].update(field_array(handle, field, record['sample'], slice(start, min(start + 16, config['frames']))))
            for field in config.get('static_fields', []):
                moments[field['name']].update(field_array(handle, field, record['sample']))
        print(json.dumps(dict(event='normalization', case=record['case_id'])), flush=True)
    inspected.update(source_digest=signature, normalization={name: value.result() for name, value in moments.items()},
                     parameter_statistics=parameter_statistics(inspected['records'], config), fit_split='train')
    inspected['digest'] = digest(inspected)
    atomic_json(path, inspected)
    return inspected


def artifact_root(config, out):
    source = config.get('artifact_source')
    if source is None:
        return Path(out)
    if not source.get('code') or not source.get('root'):
        raise ValueError('Artifact reuse requires an explicit root and producer code signature')
    root = Path(source['root']).resolve()
    if root == Path(out).resolve():
        raise ValueError('Reused artifacts must be separate from experiment outputs')
    return root


def load_prepared(config, out):
    prepared = json.loads((artifact_root(config, out) / 'prepared.json').read_text())
    fingerprint = prepared.pop('digest')
    if digest(prepared) != fingerprint or data_configuration(prepared['config']) != data_configuration(config):
        raise ValueError('Preparation/configuration mismatch')
    prepared['digest'] = fingerprint
    for value in prepared['artifacts'].values():
        verify_identity(value)
    return prepared


def data_configuration(config):
    selected = {key: config[key] for key in ('version', 'dataset', 'data_root', 'directory', 'splits', 'shape',
                                            'frames', 'dynamic_fields', 'parameters')}
    selected['static_fields'] = config.get('static_fields', [])
    selected['bins'] = config['conditioning']['bins']
    return selected


def read_frame(record, fields, timestep=None):
    with h5py.File(record['file'], 'r') as handle:
        return np.stack([field_array(handle, field, record['sample'], timestep) for field in fields])
