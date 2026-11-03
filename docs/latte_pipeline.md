# Config-driven scalar-conditioned forecasting

The pipeline learns independent direct forecasts from the initial physical fields, named numerical controls and requested physical time. Autoregression is over spatial tokens within each forecast, not over successive physical timesteps. Four supplied configurations cover Well-style HDF5 scalar/vector fields and optional static material maps at native resolution.

## Installation and configuration

**For a new dataset that already matches our supported 2D, fixed-grid Well-style HDF5 contract, adding/editing a YAML config and setting local paths is sufficient: reuse the existing preparation, training, and evaluation scripts without writing a new loader or training loop.** The config declares field/scalar paths, shape, frame/split counts, conditioning, and training options. Not every Well-format dataset meets this subset: different layouts, 3D/unstructured grids, or time-varying controls require an adapter. See the [exact contract and recipe](latte_pipeline_new_dataset.md).

Use Python 3.11 and install a hardware-compatible PyTorch 2.5+ build, then `python -m pip install -r requirements-pipeline.txt`. The CPU regression environment uses PyTorch 2.7.1. The licensed pinned Phaedra implementation must be present under `third_party/Phaedra`; its MIT license and upstream commit are retained. Run commands from the checkout root and set `PYTHONPATH` to that root.

```bash
python -m unittest scripts.test_latte_pipeline scripts.test_latte_ablation_sweep scripts.test_package_latte_pipeline -v
```

These synthetic CPU tests need no private data, pretrained weights, Slurm, or GPU. They exercise both conditioning encoders, the no-parameter control, tiny training/evaluation, checkpoint migration, exact interrupted-training resume, dependency gating, and release packaging. Synthetic correctness is not a real-data accuracy claim. The GitHub workflow also runs the tests from the isolated release bundle to catch undeclared local-file dependencies.

For another dataset, follow [the HDF5 contract and configuration recipe](latte_pipeline_new_dataset.md). The pipeline consumes named numerical scalars already extracted from input decks; it does not parse arbitrary deck languages or encode free text.

Set these environment variables to your own paths:

```bash
export LATTE_DATA_ROOT=/path/to/dataset-parent
export LATTE_WORK_ROOT=/path/to/new-experiment
export LATTE_TOKENIZER_CHECKPOINT=/path/to/original-df8-best-ema.pt
export LATTE_TRANSFORMER_CHECKPOINT=/path/to/source-ar583-step140000.pt
export LATTE_PYTHON=/path/to/environment/bin/python
```

Data and model weights are **not included**. Fine-tuning requires compatible LATTE source checkpoints, not an arbitrary upstream transformer: the supplied recipe expects the original Phaedra df8 EMA step98,000 tokenizer and the compatible AR583M source model. No public download location for those LATTE weights is promised by this code release. Scratch trains its own tokenizer and transformer, but the common YAML still requires all environment variables to resolve; unused pretrained paths may point to nonexistent placeholders for scratch-only runs. Only load checkpoints from trusted sources: training resumes use PyTorch pickle-based loading.

Each YAML declares shard directories, train/valid counts, native shape, number of frames, named fields and vector component indices, physical grouping for metrics, scalar paths, units (or `unknown`), conditioning and training settings. Physical data must have sample/time/x/y axes, with a trailing component axis for vector/tensor components; static fields omit time. Shared coordinate/time arrays live under `dimensions/`. Case IDs are read solely for indexing/auditing, not model conditioning. Unknown formats fail explicitly; a config is not a general-purpose arbitrary-file parser.

```bash
python scripts/latte_pipeline.py inspect --config configs/pde_sidd/cloverleaf.yaml
python scripts/latte_pipeline.py prepare --config configs/pde_sidd/cloverleaf.yaml
python scripts/latte_pipeline.py submit --config configs/pde_sidd/cloverleaf.yaml \
  --arm scratch --account YOUR_ACCOUNT
python scripts/latte_pipeline.py submit --config configs/pde_sidd/cloverleaf.yaml \
  --arm finetune --account YOUR_ACCOUNT
```

`submit` targets the supplied single-node eight-H100 Slurm profile and launches CPU-only stage watchers. Sites with other hardware must adapt the resource profile rather than assume H100 resource names exist. No concurrency cap is imposed. Each stage has an eight-hour allocation; only TIMEOUT retries the same stage, COMPLETED advances, other failures stop. Duplicate submissions are rejected. Workflows must use a pinned checkout. Logs and exact scheduler IDs live in each dataset output directory.

Individual GPU commands are `tokenizer-preflight`, `train-tokenizer` (scratch only), `cache`, `preflight`, `train`, and `evaluate`, with the same `--config`/`--arm` interface. Use the supplied batch launcher; never overwrite allocation GPU masks with physical GPU guesses. Preparation uses an exclusive lock shared by both arms. Training preflights perform disposable real updates, never overwrite selected weights, and bind code/config/cache identities.

## Conditioning and frozen token caches

```yaml
conditioning:
  encoding: binned
  bins: 64
```

For every scalar, all training trajectories determine min/max, mean/std, quantiles, unique counts, bin occupancy and collision counts. Validation does not fit anything. Bins use nearest-even rounding to the64 equally spaced training-range levels, matching the old numerical token recipe; missing/underflow/overflow have separate states. Distinct outside-range values share an overflow state. No claim of faithful numerical extrapolation is made for binned conditioning. Required controls must be finite in these four datasets. Constant controls have one in-range token.

Set `encoding: continuous` to use parameter identity plus a learned two-layer value/availability encoder. Values are train-standardized without clipping. The token cache and physical preprocessing are unchanged when only encoding changes; forecast outputs live in separate `scratch/binned`, `scratch/continuous`, etc. Run the new forecast `preflight` and `train` stages; do not reuse a binned transformer checkpoint as a continuous-model resume. Physical time always has a separate continuous token, normalized by the common training time span. No language model or text caption generation is involved.

The scratch arm first trains its own full df8 tokenizer, then freezes it. The fine-tuning arm always uses the supplied frozen PDEGym6 EMA98k tokenizer. The comparison includes tokenizer differences; it does not isolate transformer pretraining. Cached tokens preserve separate8640-class morphology and1024-class amplitude families. Original transformer token/special IDs never shift. New input controls use separate embeddings, and new fields have explicit identities. For256² inputs, pretrained16-position row/column embeddings are interpolated to32 positions, not input fields resized to128².

## Results and safety

### Conditioning ablations after a core sweep

`conditioning.encoding: none` removes **parameter tokens**, not the observed physical fields, static material maps, dataset token, or continuous prediction-time token. No parameter encoder is constructed; changing the scalar values cannot change this model's outputs. Acoustics still receives the material maps, which may encode information related to its scalar controls. This is a scalar-token ablation, not removal of every physical clue about the parameters.

To compare the core binned experiments against no-parameter and continuous embeddings, register sixteen additional forecasting runs (four datasets × two arms × two encodings):

```bash
python scripts/latte_ablation_sweep.py prepare \
  --core-root /path/to/core-sweep --out /path/to/separate-ablation-sweep --account YOUR_ACCOUNT
python scripts/latte_ablation_sweep.py submit --manifest /path/to/separate-ablation-sweep/sweep.json
```

Use the same `LATTE_*` environment values as the core configurations when preparing. Generated configs are fully resolved snapshots, and their digests are checked before submission. The output root must be separate from the core tree. Each encoding has its own workflow, logs, checkpoints and validation directory. Fine-tuning starts again from the original source transformer; scratch starts from random transformer weights. Neither starts from a trained binned transformer. Hyperparameters, random seed, data, normalization, selected tokenizers, and cached tokens are otherwise reused from the corresponding core arm. No extra tokenizer training or token re-export is submitted.

The reuse config explicitly records `artifact_source.root` and its producer `code` signature. Reads verify the preparation, source identities, cache coverage, field schema, geometry and tokenizer identity. Forecast checkpoints bind both the original cache fingerprint and the new code/configuration. Cache/preparation/tokenizer-writing commands are forbidden for these reuse configurations.

Each initial eight-GPU preflight has an `afterok` dependency on a held, CPU-only barrier. Short CPU-only monitor jobs depend on the current core continuation jobs and follow their replacement job IDs across timeouts. Only after **all eight core workflows complete training and full validation** does a monitor release the barrier; the barrier independently rechecks completion. Every ablation stage also checks the prerequisite manifest. Core failures stop the release chain rather than launching ablations. Never manually release the barrier to bypass a failed core experiment. The monitor does not occupy GPUs or poll continuously. Subsequent stages retain timeout-only continuation, eight-hour allocations, and generated-validation stopping.

Check `sweep.json` for the barrier/monitor IDs and each experiment's `workflow_*.json` for current jobs. The scheduler labels ablation GPU jobs with `none` or `continuous`. Synthetic tests for both the pipeline and dependency gate run with `python -m unittest scripts.test_latte_pipeline scripts.test_latte_ablation_sweep`.

Full distributed evaluation uses an eight-hour collective timeout so ranks that resume fewer unfinished trajectories can wait for slower ranks within the allocation. This does not alter model weights, normalization, predictions, or the training stopping rule.

`prepared.json` contains source identities, train-only field normalization, parameter statistics and native coordinates. Each cache binds this manifest and the selected tokenizer hash. Checkpoints restore optimizer, EMA where applicable, per-rank RNG and deterministic global draw position. Code/data/config/world-size changes fail closed. Effective batches use accumulation, not an implicit dataset epoch definition.

The generated validation monitor selects model checkpoints on a fixed16-case/five-horizon panel. Final `evaluate` covers every validation trajectory and future time, saving per-field MSE/L1/gradient error/absolute mean bias, grouped curves, token usage/perplexity, selected sample predictions and summary JSON. Velocity components are averaged first, then physical groups equally. RMSE is derived from aggregated MSE. Absolute mean bias is not a geometry-weighted conservation integral. Native output clipping matches the tokenizer preprocessing bounds; targets retain raw stored physical values. Reconstruction is reported separately and is not a forecasting error floor.

This version intentionally neither traverses protected test directories nor exports blind-test predictions. No test-derived calibration, outcome metadata, dataset files, checkpoint weights or credentials belong in a public release. Public release is gated on clean-install tests, all four native-data preflights and a completed real scratch/fine-tuning pair.

## Reviewable public update

`python scripts/package_latte_pipeline.py --destination /path/to/new-bundle` stages only the portable allowlist, retaining dependency licensing and writing SHA256 hashes. It never copies private experiment ledgers, data, model weights, or Git history. The automated private-path/credential checks are a guardrail, not a substitute for reviewing every file.

For the real-run gate, add `--require-release-gate --work-root /path/to/core-sweep`. This requires all eight native eight-GPU preflights plus one dataset with both arms' completed training and full 100-case/all-horizon validation, without test data. A passed real-run gate alone does not mark the bundle publication-approved. Fresh-environment tests, target-repository diff/license review, and maintainer approval remain separate checks. Prepare a focused PR against the public repository; never merge private repository history or replace its README/license wholesale.
