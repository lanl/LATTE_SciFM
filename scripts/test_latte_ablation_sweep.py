import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import latte_ablation_sweep as sweep
from scripts.latte_pipeline import setup_evaluation
from src.latte_pipeline.data import atomic_json
from src.latte_pipeline import scheduler


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = dict(core=[], barrier_job='900', registered=True, released=False)
        for dataset in sweep.DATASETS:
            for arm in sweep.ARMS:
                path = self.root / dataset / f'workflow_{arm}.json'
                job = str(100 + len(self.manifest['core']) * 2)
                atomic_json(path, dict(configuration_digest='config', out=str(path.parent), arm=arm,
                    active_job=job, history=[dict(job=job, watcher=str(int(job) + 1))]))
                self.manifest['core'].append(dict(workflow=str(path), configuration_digest='config', validation_cases=100))

    def complete_core(self):
        for entry in self.manifest['core']:
            path = Path(entry['workflow'])
            workflow = json.loads(path.read_text())
            workflow.update(complete=True, active_job=None)
            atomic_json(path, workflow)
            run = path.parent / workflow['arm'] / 'binned'
            atomic_json(run / 'completion.json', dict(complete=True))
            atomic_json(run / 'validation/summary.json', dict(complete=True, cases=100))

    def test_entire_sweep_required_and_failures_block(self):
        self.assertEqual(len(sweep.core_status(self.manifest)), 8)
        with self.assertRaisesRegex(RuntimeError, 'entire core sweep'):
            sweep.require_core_complete(self.manifest)
        self.complete_core()
        sweep.require_core_complete(self.manifest)
        entry = self.manifest['core'][0]
        path = Path(entry['workflow'])
        workflow = json.loads(path.read_text())
        workflow.update(stopped=True, reason='FAILED')
        atomic_json(path, workflow)
        with self.assertRaisesRegex(RuntimeError, 'Core workflow stopped'):
            sweep.require_core_complete(self.manifest)

    def test_validation_must_be_complete(self):
        self.complete_core()
        path = self.root / sweep.DATASETS[0] / sweep.ARMS[0] / 'binned/validation/summary.json'
        atomic_json(path, dict(complete=True, cases=99))
        with self.assertRaisesRegex(ValueError, 'full validation incomplete'):
            sweep.require_core_complete(self.manifest)

    def test_controller_follows_new_watchers_and_releases_only_when_ready(self):
        path = self.root / 'sweep.json'
        with patch.object(sweep, 'cpu_job', return_value='901') as cpu, patch.object(sweep.subprocess, 'run') as release:
            sweep.advance_gate(path, self.manifest)
            cpu.assert_called_once()
            release.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'did not advance'):
                sweep.advance_gate(path, self.manifest)
            for entry in self.manifest['core']:
                workflow_path = Path(entry['workflow'])
                workflow = json.loads(workflow_path.read_text())
                workflow['active_job'] = str(int(workflow['active_job']) + 1000)
                workflow['history'].append(dict(job=workflow['active_job'], watcher=str(int(workflow['active_job']) + 1)))
                atomic_json(workflow_path, workflow)
            sweep.advance_gate(path, self.manifest)
            self.assertEqual(cpu.call_count, 2)
            release.assert_not_called()
            self.complete_core()
            with patch.object(sweep.subprocess, 'check_output', return_value='JobState=PENDING Reason=JobHeldUser'):
                sweep.advance_gate(path, self.manifest)
            release.assert_called_once_with(['scontrol', 'release', '900'], check=True)
            self.assertTrue(self.manifest['released'])

    def test_dependency_and_cpu_only_gate_resources(self):
        self.manifest.update(partition='standard', account='fixture', python='python', repo=str(self.root))
        with patch.object(sweep.subprocess, 'check_output', return_value='902\n') as command:
            self.assertEqual(sweep.cpu_job(self.manifest, self.root / 'sweep.json', 'monitor', ['101', '103']), '902')
            arguments = command.call_args.args[0]
            self.assertIn('--dependency=afterany:101:103', arguments)
            self.assertIn('--gres=none', arguments)
            self.assertIn('--cpus-per-task=1', arguments)

    def test_ablation_jobs_depend_on_barrier_and_retry_without_stale_dependency(self):
        workflow = dict(repo=str(self.root), stages=['preflight', 'train', 'evaluate'], index=0,
            out=str(self.root / 'ablation'), partition='standard', account='fixture', config='config.yaml',
            arm='scratch', python='python', dataset='fixture', encoding='none', reuse_artifacts=True, after_job='900')
        with patch.object(scheduler.subprocess, 'check_output', side_effect=['910\n', '911\n', '912\n', '913\n']) as command:
            scheduler.launch(self.root / 'workflow.json', workflow)
            self.assertIn('--dependency=afterok:900', command.call_args_list[0].args[0])
            self.assertIn('latte-fixture-scratch-none-preflight', command.call_args_list[0].args[0])
            scheduler.launch(self.root / 'workflow.json', workflow)
            self.assertNotIn('--dependency=afterok:900', command.call_args_list[2].args[0])

    def test_evaluation_timeout_is_allocation_sized(self):
        with patch.dict(os.environ, {'RANK': '1', 'WORLD_SIZE': '2'}), patch('torch.cuda.is_available', return_value=False), \
                patch('torch.distributed.init_process_group') as initialize:
            rank, world, device = setup_evaluation()
            self.assertEqual((rank, world, str(device)), (1, 2, 'cpu'))
            initialize.assert_called_once_with('gloo', timeout=datetime.timedelta(hours=8))

    def test_prepare_and_register_all_sixteen_isolated_runs(self):
        from src.latte_pipeline.data import digest, read_config
        environment = dict(LATTE_DATA_ROOT=str(self.root / 'data'), LATTE_WORK_ROOT=str(self.root),
                           LATTE_TOKENIZER_CHECKPOINT=str(self.root / 'tokenizer.pt'),
                           LATTE_TRANSFORMER_CHECKPOINT=str(self.root / 'transformer.pt'))
        with patch.dict(os.environ, environment):
            for dataset in sweep.DATASETS:
                config_path = sweep.ROOT / 'configs/pde_sidd' / f'{dataset}.yaml'
                config = read_config(config_path)
                for arm in sweep.ARMS:
                    path = self.root / dataset / f'workflow_{arm}.json'
                    workflow = json.loads(path.read_text())
                    workflow.update(dataset=dataset, config=str(config_path), configuration_digest=digest(config))
                    atomic_json(path, workflow)
                    atomic_json(path.parent / 'cache' / arm / 'metadata.json', dict(code='original-producer'))
            output = self.root.parent / (self.root.name + '-ablations')
            self.addCleanup(shutil.rmtree, output, True)
            with patch.object(sweep.subprocess, 'check_output', return_value='fixture-revision\n'):
                sweep.prepare(self.root, output, 'fixture', 'standard')
            path = output / 'sweep.json'
            manifest = json.loads(path.read_text())
            self.assertEqual(len(manifest['experiments']), 16)
            with patch.object(sweep.subprocess, 'check_output', return_value='fixture-revision\n'):
                sweep.verify_manifest(manifest)
            with patch.object(sweep, 'cpu_job', side_effect=['900', '901']), patch.object(sweep, 'submit') as register:
                sweep.register(path, manifest)
                self.assertEqual(register.call_count, 16)
                for call in register.call_args_list:
                    config = call.args[0]
                    self.assertIn(config['conditioning']['encoding'], ('none', 'continuous'))
                    self.assertEqual(config['artifact_source']['code'], 'original-producer')
                    self.assertEqual(call.args[-1], '900')
                sweep.register(path, manifest)
                self.assertEqual(register.call_count, 16)


if __name__ == '__main__':
    unittest.main()
