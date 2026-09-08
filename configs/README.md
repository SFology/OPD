# 配置索引

## `experiments/`

训练配置由 `scripts/run_opd_experiment.py` 解析，支持通过 `extends` 继承。当前正式对比使用：

- `opd_dense_discrete_lcb_base.yaml`：OPD/LCB-OPD 公共配置；
- `opd_dense_discrete_lcb_opd{,_probe}.yaml`：只测量 LCB、使用原始 OPD reward 训练；
- `opd_dense_discrete_lcb_treatment{,_probe}.yaml`：使用 LCB 缩放后的 reward 训练。

`opd_default.yaml`、`opd_rtx6000_*` 和 `opd_completion_first_*` 保留用于原始 OPD 复现、硬件探测和历史
checkpoint 兼容。旧的 hard-min ROPD 配置已经移除，不能与当前 LCB 方案混用。

## `trustworthy_opd/`

推理期教师可靠性实验配置，包括 pilot、paired-content、四组 continuation、PPL 分支比较、统一解码和
locality ablation。它们不通过 managed training launcher 运行，入口见
`scripts/trustworthy_opd/README.md`。

## 修改原则

- 已经落盘的 run 使用其目录内的 resolved `config.yaml`，不要回改；
- 新实验通过新增或复制配置表达，不覆盖已完成实验；
- 正式 OPD/ROPD 对照除目标开关外保持所有参数一致；
- 长实验先 `--dry-run` 或 probe，再交由用户通过 tmux 启动。
