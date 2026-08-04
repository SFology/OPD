# OPD experiment management

Managed runs use one immutable `RUN_ID` for configuration, logs, checkpoints,
rollouts, validation output, evaluation output, and SwanLab records. Runtime
artifacts live under `${OPD_STORAGE_ROOT}/experiments`; only launch code and
source configurations are committed to Git.

## Files in one run

```text
<RUN_ID>/
├── config.yaml                 # fully resolved source-of-truth parameters
├── manifest.yaml               # Git, host, data hashes, and model inventory
├── command.sh                  # exact Hydra command
├── status.yaml                 # prepared/running/completed/failed
├── environment/                # pip, conda, GPU and source diff snapshots
├── logs/train.log              # complete stdout/stderr
├── swanlab/                    # local SwanLab record
├── checkpoints/global_step_*/  # resumable actor and dataloader state
├── rollouts/                   # optional JSONL generations
├── validation/                 # validation generations
└── evaluation/                 # post-training evaluation artifacts
```

## Commands

Activate the environment and enter the repository first:

```bash
conda activate opd
cd "$OPD_ROOT"
```

Create the deterministic smoke dataset once:

```bash
python scripts/prepare_opd_smoke_data.py
```

Validate a configuration without writing anything:

```bash
python scripts/run_opd_experiment.py configs/experiments/opd_smoke.yaml --dry-run
```

Materialize all metadata without launching a GPU job:

```bash
python scripts/run_opd_experiment.py configs/experiments/opd_smoke.yaml --prepare-only
```

This creates a `prepared` run. Launch that exact prepared run later with
`--resume-run <RUN_ID>`; with no checkpoint present, verl starts it from step 0.

Launch the smoke run after all eight GPUs are available:

```bash
python scripts/run_opd_experiment.py configs/experiments/opd_smoke.yaml
```

Launch the full default reproduction:

```bash
python scripts/run_opd_experiment.py configs/experiments/opd_default.yaml
```

Override a value without editing the canonical config:

```bash
python scripts/run_opd_experiment.py configs/experiments/opd_default.yaml \
  --set distillation.log_prob_top_k=8 \
  --set experiment.name=opd_topk8
```

Resume an interrupted managed run from its latest checkpoint:

```bash
python scripts/run_opd_experiment.py --resume-run <RUN_ID>
```

List local runs:

```bash
python scripts/list_opd_runs.py
```

## Policy

- Never edit a generated run's `config.yaml`; use a new run for a new setup.
- Use `--resume-run` only to continue the same setup.
- Commit launcher/config changes before a formal run. Dirty state and tracked
  diffs are recorded, but a clean commit remains the strongest provenance.
- Keep SwanLab as a visualization/index service; the local run directory is the
  durable source of truth.
- Full rollout dumping is enabled for smoke/analysis runs and disabled for the
  default full run to control storage usage.
- Checkpoints are saved every 20 optimizer steps in the default run and only the
  latest three actor checkpoints are retained.
