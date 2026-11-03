import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import package_latte_pipeline as package


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def evidence(self):
        for dataset in ('acoustics', 'cloverleaf', 'gray_scott', 'kolmogorov'):
            for arm in ('scratch', 'finetune'):
                folder = self.root / dataset / arm / 'binned'
                folder.mkdir(parents=True)
                (folder / 'preflight.json').write_text(json.dumps(dict(passed=True,
                    ranks=[dict(rank=rank, loss=1.) for rank in range(8)])))
                if dataset == 'kolmogorov':
                    (folder / 'completion.json').write_text(json.dumps(dict(complete=True)))
                    (folder / 'validation').mkdir()
                    (folder / 'validation/summary.json').write_text(json.dumps(dict(complete=True,
                        dataset=dataset, arm=arm, encoding='binned', cases=100, horizons=60,
                        times=list(range(1, 61)), test_used=False)))

    def test_requires_work_root(self):
        with self.assertRaisesRegex(ValueError, 'work-root'):
            package.release_gate(None)

    def test_complete_pair(self):
        self.evidence()
        self.assertEqual(package.release_gate(self.root), ['kolmogorov'])

    def test_bad_or_duplicate_ranks_block(self):
        self.evidence()
        path = self.root / 'acoustics/scratch/binned/preflight.json'
        for ranks in ([dict(rank=0, loss=1.)] * 8,
                      [dict(rank=rank, loss=float('nan')) for rank in range(8)]):
            path.write_text(json.dumps(dict(passed=True, ranks=ranks)))
            with self.assertRaisesRegex(ValueError, 'preflight'):
                package.release_gate(self.root)

    def test_incomplete_or_wrong_validation_blocks(self):
        self.evidence()
        path = self.root / 'kolmogorov/scratch/binned/validation/summary.json'
        report = json.loads(path.read_text())
        for key, value in [('complete', False), ('cases', 99), ('test_used', True),
                           ('horizons', 59), ('times', []), ('arm', 'finetune'), ('dataset', 'acoustics')]:
            with self.subTest(key=key):
                path.write_text(json.dumps(dict(report, **{key: value})))
                with self.assertRaisesRegex(ValueError, 'complete real'):
                    package.release_gate(self.root)
        path.write_text(json.dumps(report))
        (path.parent.parent / 'completion.json').unlink()
        with self.assertRaisesRegex(ValueError, 'complete real'):
            package.release_gate(self.root)

    def test_staging_hashes_allowlist_and_preserves_existing(self):
        output = self.root / 'bundle'
        manifest = package.stage(output)
        self.assertFalse(manifest['publication_gate_passed'])
        self.assertFalse(manifest['real_run_gate_passed'])
        for name, expected in manifest['files'].items():
            self.assertEqual(package.file_hash(output / name), expected)
            self.assertNotIn(name, ['docs/paper/cast_experiment_ledger.md', 'AGENTS.md'])
        with self.assertRaises(FileExistsError):
            package.stage(output)

    def test_private_paths_credentials_and_symlinks_leave_no_bundle(self):
        source = self.root / 'source'
        source.mkdir()
        file = source / 'example.txt'
        output = self.root / 'bundle'
        for content in ('/home/' + 'private-user/data', 'ghp_' + 'a' * 36):
            file.write_text(content)
            with patch.object(package, 'ROOT', source), patch.object(package, 'release_files', return_value=[file]):
                with self.assertRaises(ValueError):
                    package.stage(output)
            self.assertFalse(output.exists())
        link = source / 'linked.txt'
        link.symlink_to(file)
        with patch.object(package, 'ROOT', source), patch.object(package, 'release_files', return_value=[link]):
            with self.assertRaisesRegex(ValueError, 'Invalid release'):
                package.stage(output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
