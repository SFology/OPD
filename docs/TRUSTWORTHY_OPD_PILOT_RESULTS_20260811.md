# Trustworthy OPD pilot results: 2026-08-11

Run directory:
`/attached/remote-home1/liufengkai/opd/trustworthy_opd/20260811_094521_trustworthy_opd_pilot`

## Data quality

- 128 student trajectories and 512 sampled states;
- 97/128 trajectories (75.8%) reached the 7168-token limit;
- 35/128 trajectories (27.3%) had a parseable final answer;
- 22/128 trajectories (17.2%) were correct;
- every naturally terminated trajectory was parseable, versus only 4.1% of
  truncated trajectories, identifying truncation rather than the grader as the
  primary collection failure.

For held-out task labels, student continuations were 61.5% parseable and 41.7%
truncated; teacher continuations were 81.2% parseable and 18.8% truncated.
Mean success was `Q_S=0.198` and `Q_T=0.458` over 24 anchor states.

## Metric signal

On `dual_knn`, independently sampled teacher self-consistency and semantic
entropy were the strongest predictors of teacher failure: AUROC 0.90 and
absolute Spearman correlation about 0.74. Conservative coverage-adjusted
versions computed retrospectively remained strong (AUROC 0.84-0.85), showing
that the result is not explained only by excluding unparseable outputs.

Teacher prefix PPL showed a moderate signal (AUROC 0.73; Spearman -0.40,
`p=0.055`). Full-vocabulary KL/JSD reached AUROC 0.64-0.65. Teacher entropy,
maximum probability, sampled-action probability, and top-1 agreement were near
chance.

Under the preregistered `mid_tail_mean_8 + dual_knn` configuration, the proposed
teacher maximum had Spearman `+0.36` and unreliable-state AUROC `0.24` when its
declared direction treated larger values as greater risk. The proposed relative
maximum had Spearman `+0.53` against `Q_T` and AUROC `0.20`; against `Q_T-Q_S`
it had Spearman `+0.16` and AUROC `0.37`. These signs do not support the initial
directional hypothesis. This is evidence against the current
neighborhood/metric formulation, not yet a definitive rejection.

## Limits and decision

- Only 24 states were validated, with four held-out continuations per role, so
  task-success labels are coarse.
- Anchor selection was stratified by `relative_max`, which can bias comparisons
  involving that metric.
- State-level baselines repeat across representations; identical scores across
  representation rows do not imply that an embedding representation is best.
- High truncation confounds trajectory coverage and continuation success.

The next experiment is therefore a truncation preflight, not a larger metric
run. It raises the budget to 16384 tokens on eight prompts and requires at least
75% parseable trajectories with at most 25% truncation before downstream work
is allowed.
