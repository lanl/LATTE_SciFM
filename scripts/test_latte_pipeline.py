import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
import yaml

from src.latte_pipeline import data, runtime
from src.latte_pipeline.conditioning import ParameterEncoder
from src.latte_pipeline.model import ForecastModel, flatten_codes, unflatten_codes, layout, token_loss, BOS
from src.latte_pipeline.scheduler import stages, terminal_action
from scripts.latte_pipeline_gpu import allocated_device
from scripts.llama_cached_inference import CachedDecoder
from scripts.train_encdec_vq_transformer_pdegym6_llama import EncDecSharedGPT
from src.latte_pipeline import model as model_module


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(version=1, dataset='fixture', directory='fixture', data_root=str(self.root), work_root=str(self.root / 'runs'),
            shape=[16, 24], frames=3, splits=dict(train=2, valid=1),
            dynamic_fields=[dict(name='pressure', path='t0_fields/pressure', semantic='pressure', group='pressure', unit='unknown')],
            static_fields=[dict(name='material', path='t0_fields/material', group='material', unit='unknown')],
            parameters=[dict(name='coefficient', path='scalars/coefficient', unit='unknown')],
            conditioning=dict(encoding='binned', bins=64), scheduler=dict(gpus=1, hours=8),
            training=dict(seed=7, global_batch=2, max_steps=2, save_every=1, eval_every=1, patience=10,
                          relative_improvement=.001, warmup=1, tokenizer_lr=1e-4, scratch_lr=1e-4,
                          finetune_lr=1e-5, dropout=0., monitor_cases=1),
            development_architecture=dict(n_layer_enc=1, n_layer_dec=1, n_head=4, n_embd=32, mlp_hidden_dim=64))
        tokenizer_config = yaml.safe_load(Path('configs/phaedra_pdegym6_df8_reference.yaml').read_text())
        tokenizer_config['tokenizer_hyperparameters']['vae_hyperparameters'].update(latent_channels=32, num_res_blocks=1,
            encoder_channel_mult=[1, 1, 1, 1], decoder_channel_mult=[1, 1, 1, 1], attn_resolutions=[])
        self.config['tokenizer_config'] = str(self.root / 'tokenizer.yaml')
        Path(self.config['tokenizer_config']).write_text(yaml.safe_dump(tokenizer_config))
        self.config['pretrained_tokenizer'] = str(self.root / 'unused.pt')
        self.config['pretrained_transformer'] = str(self.root / 'unused_transformer.pt')
        for split, count in self.config['splits'].items():
            folder = self.root / 'fixture/data' / split
            folder.mkdir(parents=True)
            with h5py.File(folder / 'shard.h5', 'w') as handle:
                handle['provenance/case_id'] = np.asarray([f'{split}_{index}'.encode() for index in range(count)])
                handle['dimensions/time'] = [0., .5, 1.]
                handle['dimensions/x'] = np.arange(16)
                handle['dimensions/y'] = np.arange(24)
                handle['scalars/coefficient'] = np.arange(count) + (1 if split == 'train' else 99)
                handle['scalars/coefficient'].attrs.update(sample_varying=True, time_varying=False)
                generator = np.random.default_rng(4)
                handle['t0_fields/pressure'] = generator.normal(size=(count, 3, 16, 24)) + (0 if split == 'train' else 10)
                handle['t0_fields/material'] = generator.normal(size=(count, 16, 24))
                for name, varying in [('pressure', True), ('material', False)]:
                    handle['t0_fields/' + name].attrs.update(sample_varying=True, time_varying=varying)
        self.out = self.root / 'runs/fixture'

    def prepare(self):
        return data.prepare(self.config, self.out)

    def test_training_only_ranges_and_static_roles(self):
        prepared = self.prepare()
        self.assertEqual(prepared['parameter_statistics'][0]['maximum'], 2)
        self.assertLess(prepared['normalization']['pressure']['physical_mean'], 1)
        self.assertEqual(prepared['normalization']['material']['count'], 2 * 16 * 24)
        changed = copy.deepcopy(self.config)
        changed['conditioning']['encoding'] = 'continuous'
        self.assertEqual(data.load_prepared(changed, self.out)['digest'], prepared['digest'])
        changed['shape'] = [128, 128]
        with self.assertRaises(ValueError):
            data.load_prepared(changed, self.out)
        with self.assertRaises(ValueError):
            data.read_frame(prepared['records'][0], self.config['static_fields'], 0)

    def test_disjoint_case_ids_and_shapes(self):
        path = self.root / 'fixture/data/valid/shard.h5'
        with h5py.File(path, 'a') as handle:
            handle['provenance/case_id'][0] = b'train_0'
        with self.assertRaises(ValueError):
            data.inspect(self.config)

    def test_binned_and_continuous_values(self):
        statistics = [dict(minimum=1., maximum=2., mean=1.5, scale=.5)]
        values = np.asarray([np.nan, 0., 1., 1.5, 2., 3.])
        expected = data.bin_values(values, statistics[0], 64)
        encoder = ParameterEncoder(statistics, 16)
        np.testing.assert_array_equal(encoder.token_indices(torch.tensor(values[:, None])).numpy()[:, 0], expected)
        self.assertEqual(expected.tolist(), [0, 65, 1, 33, 64, 66])
        constant = ParameterEncoder([dict(minimum=2., maximum=2., mean=2., scale=1.)], 16)
        self.assertEqual(constant.token_indices(torch.tensor([[2.]])).item(), 1)
        continuous = ParameterEncoder(statistics, 16, 'continuous')
        inputs = torch.tensor([[2.], [3.]], dtype=torch.float64)
        outputs = continuous(inputs)
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        outputs.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in continuous.parameters()))
        self.assertTrue(torch.isfinite(continuous(torch.tensor([[float('nan')]]))).all())

    def test_prefix_and_cached_decoder_match(self):
        prepared = self.prepare()
        model = ForecastModel(self.config, prepared['parameter_statistics'], [0, .5, 1], self.config['development_architecture'])
        source_codes = torch.zeros((1, 2, 2, 2, 3), dtype=torch.long)
        target_codes = torch.zeros((1, 1, 2, 2, 3), dtype=torch.long)
        source, target = flatten_codes(source_codes), flatten_codes(target_codes)
        torch.testing.assert_close(unflatten_codes(target, 1, 2, 3), target_codes)
        controls, initial, final = torch.tensor([[1.]], dtype=torch.float64), torch.tensor([0.]), torch.tensor([1.])
        model.train()
        loss = token_loss(model(source, controls, initial, final, target), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.prefix.parameters_encoder.values.weight.grad)
        model.eval()
        with torch.no_grad():
            memory = model.encode_inputs(source, controls, initial, final)
            rows, cols, fields = layout(model.output_slots, 2, 3, source.device, 1)
            decoder = CachedDecoder(model, memory, rows, cols, fields)
            prefix = torch.full((1, 1), BOS)
            for position in range(6):
                cached = decoder.step(prefix[:, -1:])
                reference = model.decode(prefix, rows, cols, fields, memory)[:, -1]
                torch.testing.assert_close(cached, reference, rtol=2e-4, atol=3e-4)
                prefix = torch.cat((prefix, target[:, position:position + 1]), 1)
            changed = model.encode_inputs(source, controls + 1, initial, final)
            self.assertFalse(torch.equal(memory, changed))
            generated = model.generate(source, controls, initial, final)
            self.assertTrue((generated[:, ::2] < 8640).all())
            self.assertTrue(((generated[:, 1::2] >= 8640) & (generated[:, 1::2] < 9664)).all())

    def test_no_parameters_are_not_hidden_conditioning(self):
        prepared = self.prepare()
        config = copy.deepcopy(self.config)
        config['conditioning']['encoding'] = 'none'
        model = ForecastModel(config, prepared['parameter_statistics'], [0, .5, 1], config['development_architecture'])
        model.eval()
        source = flatten_codes(torch.zeros((1, 2, 2, 2, 3), dtype=torch.long))
        target = flatten_codes(torch.zeros((1, 1, 2, 2, 3), dtype=torch.long))
        controls, initial, final = torch.tensor([[1.]]), torch.tensor([0.]), torch.tensor([1.])
        self.assertIsNone(model.prefix.parameters_encoder)
        self.assertEqual(model.prefix(controls, initial, final).shape[1], 2)
        with torch.no_grad():
            reference = model(source, controls, initial, final, target)
            changed = model(source, torch.full_like(controls, float('nan')), initial, final, target)
            self.assertTrue(torch.equal(reference, changed))
            self.assertFalse(torch.equal(reference, model(source, controls, initial, final / 2, target)))
        model.train()
        token_loss(model(source, controls, initial, final, target), target).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.prefix.parameters()))

    def test_gpu_binding_and_stopping_policy(self):
        self.assertEqual(allocated_device('0,1,2,3,4,5,6,7', 6, 8), '6')
        with self.assertRaises(ValueError):
            allocated_device('0,1', 2, 8)
        self.assertEqual(terminal_action('TIMEOUT'), 'retry')
        self.assertEqual(terminal_action('FAILED'), 'stop')
        self.assertEqual(terminal_action('COMPLETED'), 'advance')
        self.assertIn('train-tokenizer', stages('scratch'))
        self.assertNotIn('train-tokenizer', stages('finetune'))
        state = dict(best=None, reference=None, stale=0)
        self.assertTrue(runtime.observe(state, .1, self.config['training']))
        for _index in range(10):
            runtime.observe(state, .1, self.config['training'])
        self.assertEqual(state['stale'], 10)

    def test_checkpoint_migration_preserves_alphabet_and_semantic_rows(self):
        prepared = self.prepare()
        architecture = self.config['development_architecture']
        source_model = EncDecSharedGPT(vocab_size=9695, n_embed=9664, Hq=16, Wq=16, C=8, deck_len=2,
            dec_block_size=2048, dropout=0., pad_id=9694, use_sdpa=True, **architecture)
        source = dict(step=140000, n_embed=9664, vocab_size=9695, Hq=16, Wq=16, C=8,
                      delta_t_choices=list(range(1, 21)), cfg=architecture, model=source_model.state_dict())
        changed = copy.deepcopy(self.config)
        changed['shape'] = [256, 256]
        target = ForecastModel(changed, prepared['parameter_statistics'], [0, .5, 1], architecture)
        with patch.object(model_module, 'ARCHITECTURE', architecture):
            report = model_module.migrate_pretrained(target, source)
        self.assertTrue(report['token_ids_unchanged'])
        torch.testing.assert_close(target.tok_emb.weight, source_model.tok_emb.weight, rtol=0, atol=0)
        torch.testing.assert_close(target.channel_emb.weight[:8], source_model.channel_emb.weight, rtol=0, atol=0)
        torch.testing.assert_close(target.row_emb.weight[[0, -1]], source_model.row_emb.weight[[0, -1]], rtol=0, atol=0)
        for encoding in ('none', 'continuous'):
            changed['conditioning']['encoding'] = encoding
            ablation = ForecastModel(changed, prepared['parameter_statistics'], [0, .5, 1], architecture)
            with patch.object(model_module, 'ARCHITECTURE', architecture):
                model_module.migrate_pretrained(ablation, source)
            torch.testing.assert_close(ablation.tok_emb.weight, source_model.tok_emb.weight, rtol=0, atol=0)
        source['model']['unexpected.weight'] = torch.zeros(1)
        with patch.object(model_module, 'ARCHITECTURE', architecture), self.assertRaises(ValueError):
            model_module.migrate_pretrained(target, source)

    def test_deterministic_distributed_sampling_and_metrics(self):
        shards = [runtime.global_draws(7, 5, 64, 100, rank, 8) for rank in range(8)]
        single = runtime.global_draws(7, 5, 64, 100, 0, 1)
        np.testing.assert_array_equal(np.stack(shards).T.reshape(-1), single)
        fields = [dict(group=group) for group in ['density', 'velocity', 'velocity', 'pressure']]
        self.assertEqual(float(runtime.grouped(np.array([3., 6., 12., 15.]), fields)), 9.)

    def test_end_to_end_scratch_and_continuous_cache_reuse(self):
        with patch.object(runtime, 'code_signature', return_value='fixture-code'):
            self.prepare()
            device = torch.device('cpu')
            runtime.tokenizer_preflight(self.config, self.out, 'scratch', 0, 1, device)
            runtime.train(self.config, self.out, 'scratch', 0, 1, device, tokenizer_training=True)
            runtime.cache(self.config, self.out, 'scratch', 0, 1, device)
            runtime.preflight(self.config, self.out, 'scratch', 0, 1, device)
            before = data.file_hash(self.out / 'tokenizer/best_ema.pt')
            runtime.train(self.config, self.out, 'scratch', 0, 1, device)
            runtime.evaluate(self.config, self.out, 'scratch', 0, 1, device)
            report = json.loads((self.out / 'scratch/binned/validation/summary.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['cases'], 1)
            self.assertFalse(report['test_used'])
            self.assertEqual(data.file_hash(self.out / 'tokenizer/best_ema.pt'), before)
            changed = copy.deepcopy(self.config)
            changed['conditioning']['encoding'] = 'continuous'
            runtime.preflight(changed, self.out, 'scratch', 0, 1, device)
            runtime.train(changed, self.out, 'scratch', 0, 1, device)
            runtime.evaluate(changed, self.out, 'scratch', 0, 1, device)
            self.assertEqual(data.file_hash(self.out / 'tokenizer/best_ema.pt'), before)
            original_files = {str(path): data.file_hash(path) for path in self.out.rglob('*') if path.is_file()}
            for encoding in ('none', 'continuous'):
                ablation = copy.deepcopy(self.config)
                ablation['conditioning']['encoding'] = encoding
                ablation['artifact_source'] = dict(root=str(self.out), code='fixture-code')
                destination = self.root / 'ablations' / encoding / 'fixture'
                ablation['work_root'] = str(destination.parent)
                runtime.preflight(ablation, destination, 'scratch', 0, 1, device)
                runtime.train(ablation, destination, 'scratch', 0, 1, device)
                runtime.evaluate(ablation, destination, 'scratch', 0, 1, device)
                result = json.loads((destination / 'scratch' / encoding / 'validation/summary.json').read_text())
                self.assertTrue(result['complete'])
                with self.assertRaisesRegex(ValueError, 'read-only'):
                    runtime.cache(ablation, destination, 'scratch', 0, 1, device)
                ablation['artifact_source']['code'] = 'wrong-producer'
                with self.assertRaisesRegex(ValueError, 'implementation mismatch'):
                    runtime.FrameCache(ablation, data.load_prepared(ablation, destination), destination, 'scratch')
            self.assertEqual(original_files, {str(path): data.file_hash(path) for path in self.out.rglob('*') if path.is_file()})
            changed['training']['scratch_lr'] *= 2
            with self.assertRaises(ValueError):
                runtime.train(changed, self.out, 'scratch', 0, 1, device)

    def test_interrupted_training_exact_resume(self):
        with patch.object(runtime, 'code_signature', return_value='fixture-code'):
            self.config['training']['dropout'] = .1
            self.prepare()
            device = torch.device('cpu')
            runtime.tokenizer_preflight(self.config, self.out, 'scratch', 0, 1, device)
            runtime.train(self.config, self.out, 'scratch', 0, 1, device, tokenizer_training=True)
            runtime.cache(self.config, self.out, 'scratch', 0, 1, device)
            runtime.preflight(self.config, self.out, 'scratch', 0, 1, device)
            real_save = runtime.atomic_checkpoint
            def interrupt(path, value):
                real_save(path, value)
                if Path(path).name == 'last.pt' and value['step'] == 1:
                    raise InterruptedError('simulated allocation interruption')
            with patch.object(runtime, 'atomic_checkpoint', side_effect=interrupt), self.assertRaises(InterruptedError):
                runtime.train(self.config, self.out, 'scratch', 0, 1, device)
            runtime.train(self.config, self.out, 'scratch', 0, 1, device)
            resumed = torch.load(self.out / 'scratch/binned/last.pt', weights_only=False)
            (self.out / 'scratch/binned').rename(self.out / 'scratch/resumed')
            runtime.preflight(self.config, self.out, 'scratch', 0, 1, device)
            runtime.train(self.config, self.out, 'scratch', 0, 1, device)
            uninterrupted = torch.load(self.out / 'scratch/binned/last.pt', weights_only=False)
            self.assertEqual(resumed['step'], uninterrupted['step'])
            for name, value in resumed['model'].items():
                torch.testing.assert_close(value, uninterrupted['model'][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
