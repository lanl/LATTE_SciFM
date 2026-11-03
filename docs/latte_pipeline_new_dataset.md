# Add a dataset without rewriting training code

**Yes: if your data already match the supported Well-style HDF5 contract below, only the dataset YAML config and your local paths need to change.** The same scripts inspect the dataset, fit training-only preprocessing, cache tokens, train or fine-tune, and evaluate. No new model, loader, or training-loop code is needed for a compatible dataset. A file being “Well-format” alone is not sufficient; check the supported dimensionality, array layout, shared grids/timestamps, and control types below.

This guide covers the **2D, fixed-grid, Well-style HDF5 adapter**, not arbitrary input-deck parsing. The supplied PDE_SIDD files already expose selected numerical input-deck values under `scalars/`. Original deck text, lists, schedules, or new file layouts require an explicit conversion/adapter first. Do not describe this release as a general text-conditioned foundation model.

## 1. Verify the on-disk contract

Place shards at `${LATTE_DATA_ROOT}/<directory>/data/train/*.h5` and `data/valid/*.h5`. Each shard contains:

| HDF5 path | Shape | Meaning |
| --- | --- | --- |
| `provenance/case_id` | `(N,)` strings | Unique trajectory IDs, disjoint across splits; never conditioning |
| `dimensions/time` | `(T,)` | Strictly increasing physical timestamps |
| `dimensions/x`, `dimensions/y` | `(Nx,)`, `(Ny,)` | Shared native coordinates |
| `t0_fields/pressure` | `(N,T,Nx,Ny)` | Example evolving scalar |
| `t1_fields/velocity` | `(N,T,Nx,Ny,2)` | Example evolving vector, explicitly select components |
| `t0_fields/material_density` | `(N,Nx,Ny)` | Example static spatial map |
| `scalars/coefficient` | `(N,)` or scalar | Per-trajectory or dataset-fixed numerical control |

Every physical field must have HDF5 attributes `sample_varying=True` and `time_varying=True` for evolving fields, or `time_varying=False` for static maps. Scalars use `sample_varying=True` for `(N,)`, or `False` for a shared scalar, and must not be time-varying. Required scalar values and physical fields must be finite. Missing-state embeddings exist, but the current data loader rejects missing required controls rather than inventing values.

All shards and splits must share the same coordinates and timestamps. Both spatial sizes must be multiples of eight; fields are not resized. The supplied real-data configurations cover 128×128 and 256×256. A new grid still needs memory and numerical preflight checks; larger rectangular grids are not a validated production claim. Choose coordinates/units consistently before ingestion; the loader does not convert units.

## 2. Copy a configuration

Save this alongside `configs/pde_sidd/common.yaml`, replacing the illustrative shape/counts/paths with verified metadata:

```yaml
defaults: common.yaml
dataset: my_simulation
directory: my_simulation
shape: [128, 128]
frames: 61
splits: {train: 800, valid: 100}
dynamic_fields:
  - {name: pressure, path: t0_fields/pressure, semantic: pressure, group: pressure, unit: unknown}
  - {name: velocity_x, path: t1_fields/velocity, component: 0, semantic: velocity_x, group: velocity, unit: unknown}
  - {name: velocity_y, path: t1_fields/velocity, component: 1, semantic: velocity_y, group: velocity, unit: unknown}
static_fields:
  - {name: material_density, path: t0_fields/material_density, group: material_density, unit: unknown}
parameters:
  - {name: coefficient, path: scalars/coefficient, unit: unknown}
conditioning: {encoding: binned, bins: 64}
```

Remove unused fields rather than filling them with zeros. Use `static_fields: []` when appropriate. Preserve names and known units; use `unknown` when unverified. Only assign pretrained `semantic` mappings when the physical meaning truly matches. A shared `group: velocity` averages component MSEs before averaging over physical groups. Static maps are observed encoder inputs, not prediction targets. They must be available at prediction time; never include future outcomes or diagnostics disguised as controls.

## 3. Inspect, prepare, and preflight

Follow the environment setup in [the pipeline guide](latte_pipeline.md), then run:

```bash
python scripts/latte_pipeline.py inspect --config configs/pde_sidd/my_simulation.yaml
python scripts/latte_pipeline.py prepare --config configs/pde_sidd/my_simulation.yaml
```

Inspect `prepared.json`: source identities, split membership, physical normalization, parameter ranges/quantiles/bin collisions, and coordinates. Only training data fit normalization and bins. Validation trajectories never fit these quantities. No protected test directory is read.

The supplied Slurm workflow automatically orders preparation, tokenizer preflight/training for scratch, frozen token caching, transformer preflight/training, and full validation. It requires a pinned checkout and the documented eight-H100 profile; adapt scheduler resources for another site before submission. Do not call `submit` merely to test configuration parsing.

```bash
python scripts/latte_pipeline.py submit --config configs/pde_sidd/my_simulation.yaml \
  --arm scratch --account YOUR_ACCOUNT --partition YOUR_PARTITION
python scripts/latte_pipeline.py submit --config configs/pde_sidd/my_simulation.yaml \
  --arm finetune --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

Use a new experiment root for changed scientific configurations. Do not edit configurations or code under running workflows. Incompatible source/cache/checkpoint signatures are intentionally rejected.

## 4. Select a conditioning experiment

- `binned`: parameter-specific discrete embeddings; training-range levels plus missing/underflow/overflow states. Different overflowing values are indistinguishable.
- `continuous`: parameter identity plus learned continuous value/availability embeddings; train-standardized, unclipped values.
- `none`: no numerical parameter tokens. Initial fields, static maps, dataset identity, and requested physical time remain available.

These are separate transformer experiments, not interchangeable resume modes. Parameter names/units describe a fixed dataset-specific schema; this version does not dynamically understand unseen parameter names or text. To compare encodings while reusing frozen artifacts safely, follow the dedicated ablation workflow in the pipeline guide.

## 5. Interpret results

Checkpoint selection uses generated forecasts on a fixed validation panel. `validation/summary.json` is the full-validation aggregate only when `complete` is true. Physical grouped MSE and train-variance-normalized grouped MSE are different quantities. A single image is not an aggregate score.

Scratch trains a dataset-specific tokenizer, whereas fine-tuning freezes the supplied original PDEGym6 tokenizer. Their comparison measures the complete pipelines, not just transformer initialization. The release does not bundle data or source checkpoints, and it does not establish that all four downstream tasks benefit from pretraining or conditioning.
