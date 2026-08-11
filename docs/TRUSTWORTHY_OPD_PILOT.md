# Trustworthy OPD inference-only pilot

This pilot tests whether local teacher instability on student-generated states
predicts real-task teacher reliability. It performs no parameter updates.
The response budget matches the original OPD setting (7168 tokens); reducing it
to a 512-token smoke-test budget truncates DeepSeek-R1-style reasoning before a
parseable final answer and makes reliability labels degenerate.

## Hypotheses

For the token `v` actually sampled by the student at state `s`, the finite-support
versions of the two proposed risks are:

```text
teacher(s, v)  = max_s' [log pi_T(v|s) - log pi_T(v|s')]
relative(s, v) = max_s' [log pi_T(v|s) - log pi_T(v|s')
                         - log pi_S(v|s) + log pi_S(v|s')]
```

Neighbors must be close in both the teacher and student representation spaces.
The experiment compares the risks with continuation success on the original
math task:

```text
Q_T(s,v) = success rate of teacher continuations from prefix s + v
Q_S(s,v) = success rate of student continuations from prefix s + v
```

The expected relationships are `teacher risk` versus `Q_T`, and `relative risk`
versus `Q_T - Q_S`.

## Representation alternatives

The pilot does not assume that a middle-layer last-token vector is the correct
state representation. Every configured representation independently produces
teacher and student neighborhoods. The initial grid includes:

- middle-layer last token;
- middle-layer mean over the last 8 tokens;
- middle-layer mean over the entire prefix;
- late-layer and final-layer last token;
- the mean of the final four layers at the last token;
- input-embedding mean over the tail or the entire prefix.

Edit `representations.definitions` in
`configs/trustworthy_opd/pilot.yaml` to add layers or pooling strategies. A
floating-point layer value is a depth fraction, `-1` is the final hidden state,
and `embedding` is the model's own input embedding output. Teacher and student
vectors are never compared across coordinate systems; each model defines its
own distance, and the two constraints are intersected.

The main neighborhood estimators are:

- `dual_knn`: intersection of each model's nearest states;
- `dual_radius`: intersection of model-specific balls whose radii are estimated
  from a configured empirical distance quantile.

Candidate states come from other rollouts of the same problem. Exact duplicate
prefixes are excluded.

## Staged execution

All GPU stages are deliberately separate and resumable. Review each command
before running it. The example below maps physical GPU 2 to `cuda:0` inside the
process.

For a complete run with automatic GPU selection, per-stage logs, collection
quality checks, and stop-on-error behavior, use the local pipeline wrapper:

```bash
bash scripts/trustworthy_opd/run_pipeline.sh
```

Resume from the status recorded in an existing run with:

```bash
bash scripts/trustworthy_opd/run_pipeline.sh --run-dir "$TRUST_OPD_RUN"
```

Use `--from-stage NAME` only when deliberately overriding automatic resume.
The wrapper reevaluates GPU availability before every GPU stage; the commands
below remain useful when running or debugging stages individually.

Set the common environment first:

```bash
conda activate opd
cd /remote-home/liufengkai/projects/OPD
export OPD_ROOT=/remote-home/liufengkai/projects/OPD
export OPD_STORAGE_ROOT=/attached/remote-home1/liufengkai/opd
export OPD_MODEL_DIR=/attached/remote-home1/liufengkai/opd/models
export CUDA_VISIBLE_DEVICES=2
```

### 1. Collect frozen student trajectories and states

```bash
python scripts/trustworthy_opd/collect_states.py \
  --config configs/trustworthy_opd/pilot.yaml
```

The command prints `RUN_DIR=...`. Save it for all later stages:

```bash
export TRUST_OPD_RUN=/attached/remote-home1/liufengkai/opd/trustworthy_opd/<RUN_ID>
```

### 2. Extract student and teacher features

Run the roles sequentially so both models are never resident on the GPU at the
same time:

```bash
python scripts/trustworthy_opd/extract_features.py \
  --run-dir "$TRUST_OPD_RUN" --role student

python scripts/trustworthy_opd/extract_features.py \
  --run-dir "$TRUST_OPD_RUN" --role teacher
```

The extractor validates that both tokenizers are identical before using a
student token ID as the teacher action. It stores all configured
representations, exact teacher/student entropy and confidence baselines,
top-k sets, and log probabilities for every sampled action occurring in the
pilot support set. With `store_full_log_probs: true`, it additionally stores a
float16 full-vocabulary log-probability matrix so teacher-student KL, reverse
KL, and Jensen-Shannon divergence use the whole distribution rather than a
top-k approximation. At the default 512 states and a roughly 152k vocabulary,
this costs about 148 MiB per model feature file.

For roles listed in `prefix_ppl_roles`, the extractor also scores each complete
student-generated trajectory once and derives prefix PPL at every sampled
state. This is more expensive than next-token feature extraction but avoids
re-scoring the same prefix separately for each state.

### 3. Build double-ball neighborhoods and stability metrics

This stage is CPU-only:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/trustworthy_opd/compute_stability.py \
  --run-dir "$TRUST_OPD_RUN"
```

In addition to the two proposed one-sided suprema, it records 95th percentile,
top-tail mean, CVaR, absolute teacher sensitivity, and relative absolute excess.
These variants reveal whether a result is driven by a single outlier neighbor.

### 4. Estimate real-task continuation reliability

The configured primary metric is used only to select a small stratified set of
anchor states. Both policies are then evaluated on exactly the same states:

```bash
export CUDA_VISIBLE_DEVICES=2

python scripts/trustworthy_opd/validate_reliability.py \
  --run-dir "$TRUST_OPD_RUN" --role student

python scripts/trustworthy_opd/validate_reliability.py \
  --run-dir "$TRUST_OPD_RUN" --role teacher
```

Each continuation begins after the actual student action `s + v`. Correctness
is graded with the repository's DAPO math verifier. Once both roles finish,
`reliability.parquet` contains `Q_T`, `Q_S`, and `Q_T-Q_S` joined to every
representation and neighborhood metric.

The first `metric_continuations_per_state` teacher generations are reserved for
self-consistency and semantic entropy. Their canonical final answers define
the semantic clusters: answer-equivalent generations share a cluster, while
unparseable answers are reported through `teacher_valid_answer_rate` and are
not merged into a false consensus. The remaining teacher and student
generations alone define `Q_T` and `Q_S`, so these metrics are not evaluated on
the same samples used to construct them.
The student only generates the held-out label half; no unused student
self-consistency samples are produced.

### 5. Analyze correlations and calibration

This stage is CPU-only:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/trustworthy_opd/analyze_results.py \
  --run-dir "$TRUST_OPD_RUN"
```

## Outputs

```text
<RUN_DIR>/
├── config.yaml
├── status.yaml
├── artifacts/
│   ├── trajectories.jsonl
│   ├── states.jsonl
│   └── selected_states.jsonl
├── features/
│   ├── student.npz
│   └── teacher.npz
└── results/
    ├── stability.parquet
    ├── stability.csv
    ├── continuations.jsonl
    ├── reliability.parquet
    ├── reliability.csv
    ├── correlations.csv
    ├── calibration.csv
    └── summary.json
```

The key comparison is whether the proposed metrics outperform these baselines
on independently estimated task success:

- teacher entropy and maximum probability;
- teacher-student full-vocabulary KL, reverse KL, and Jensen-Shannon divergence;
- teacher-student top-1 agreement and top-k overlap;
- teacher self-consistency (majority share and pairwise agreement);
- teacher semantic entropy (raw and normalized);
- teacher prefix PPL and actual student-action PPL;
- sampled-action confidence and the teacher-student action gap.

`correlations.csv` reports Spearman correlation and unreliable-state AUROC for
every candidate, while `calibration.csv` gives binned metric-versus-success
curves. This makes the comparison about predictive validity, rather than just
whether a score looks numerically well behaved.
All metrics are compared against the same primary target `Q_T`; metrics that
explicitly compare teacher and student are additionally evaluated against
`Q_T-Q_S` rather than being confined to a separate, incomparable ranking.

## Interpretation limits

- This is a finite rollout support set, so `max` estimates a support maximum,
  not the exact continuous supremum.
- Continuation success can be sparse with only four held-out samples. Treat this pilot as
  a falsification test and increase continuations only after observing a signal.
- Semantic entropy here means entropy over canonical final-answer equivalence
  classes, not an embedding-based clustering of free-form rationales. This is a
  task-aligned definition for exact-answer math, but should be replaced for
  open-ended tasks.
- Selection is stratified by one primary metric. A confirmatory run should
  sample anchors independently of all candidate metrics.
- Input-embedding pooling ignores some order information; it is included as a
  deliberately different ablation, not as a preferred representation.
