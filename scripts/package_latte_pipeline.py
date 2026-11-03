#!/usr/bin/env python3
"""Stage a portable release allowlist; require real-run gates before publication."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.latte_pipeline.data import file_hash


def release_gate(work_root):
    if work_root is None:
        raise ValueError('--work-root is required for the real-run release gate')
    work_root = Path(work_root)
    complete_pairs = []
    for dataset, horizons in [('acoustics', 60), ('cloverleaf', 60), ('gray_scott', 200), ('kolmogorov', 60)]:
        completed = []
        for arm in ('scratch', 'finetune'):
            folder = work_root / dataset / arm / 'binned'
            preflight = json.loads((folder / 'preflight.json').read_text())
            ranks = preflight.get('ranks', [])
            if (preflight.get('passed') is not True or len(ranks) != 8
                    or {row.get('rank') for row in ranks} != set(range(8))
                    or any(not math.isfinite(row.get('loss', float('nan'))) for row in ranks)):
                raise ValueError(f'Missing real eight-GPU preflight: {dataset}/{arm}')
            path = folder / 'validation/summary.json'
            if path.exists():
                report = json.loads(path.read_text())
                completion_path = folder / 'completion.json'
                completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
                if (completion.get('complete') is True and report.get('complete') is True
                        and report.get('cases') == 100 and report.get('test_used') is False
                        and report.get('dataset') == dataset and report.get('arm') == arm
                        and report.get('encoding') == 'binned' and report.get('horizons') == horizons
                        and len(report.get('times', [])) == horizons):
                    completed.append(arm)
        if len(completed) == 2:
            complete_pairs.append(dataset)
    if not complete_pairs:
        raise ValueError('Public release requires a complete real scratch/fine-tuning validation pair')
    return complete_pairs


def release_files():
    files = [*ROOT.glob('src/latte_pipeline/*.py'), *ROOT.glob('configs/pde_sidd/*.yaml'),
             *ROOT.glob('third_party/Phaedra/phaedra/*.py')]
    files += [ROOT / name for name in ['scripts/latte_pipeline.py', 'scripts/latte_pipeline_gpu.py',
        'scripts/test_latte_pipeline.py', 'scripts/package_latte_pipeline.py',
        'scripts/test_package_latte_pipeline.py', '.github/workflows/latte-pipeline.yml',
        'scripts/latte_ablation_sweep.py', 'scripts/test_latte_ablation_sweep.py',
        'scripts/train_encdec_vq_transformer_pdegym6_llama.py', 'scripts/llama_cached_inference.py',
        'src/data/resumable.py', 'sbatch/sbatch_latte_pipeline_stage.sh', 'docs/latte_pipeline.md',
        'docs/latte_pipeline_new_dataset.md',
        'requirements-pipeline.txt', 'configs/phaedra_pdegym6_df8_reference.yaml',
        'third_party/Phaedra/LICENSE', 'third_party/Phaedra/UPSTREAM_COMMIT']]
    return sorted(files)


def stage(destination, work_root=None, require_release_gate=False):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Stage into a new directory; do not overwrite a public checkout')
    pairs = release_gate(work_root) if require_release_gate else []
    files = release_files()
    for path in files:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f'Invalid release source: {path}')
        content = path.read_text()
        if re.search(r'/net/[A-Za-z0-9_-]+/exempt/|/(?:Users|users|home)/[A-Za-z0-9_.-]+/', content):
            raise ValueError(f'Private path in public allowlist: {path}')
        if re.search(r'-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b', content):
            raise ValueError(f'Potential credential in public allowlist: {path}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.latte-stage-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'bundle'
        staged.mkdir()
        hashes = {}
        for path in files:
            output = staged / path.relative_to(ROOT)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            hashes[str(path.relative_to(ROOT))] = file_hash(output)
        manifest = dict(files=hashes, real_run_gate_passed=require_release_gate, completed_pairs=pairs,
                        publication_gate_passed=False,
                        remaining_review=['clean-install tests', 'public-target diff and licensing review',
                                          'maintainer publication approval'])
        (staged / 'pipeline_release_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staged.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--require-release-gate', action='store_true', help='Require real-run evidence; not publication approval')
    parser.add_argument('--work-root', type=Path)
    args = parser.parse_args()
    if args.require_release_gate and args.work_root is None:
        parser.error('--require-release-gate requires --work-root')
    manifest = stage(args.destination, args.work_root, args.require_release_gate)
    print(f"Staged {len(manifest['files'])} portable files; real_run_gate_passed={manifest['real_run_gate_passed']}; publication requires review")


if __name__ == '__main__':
    main()
