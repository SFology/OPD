# Trustworthy On-Policy Distillation

本仓库用于复现原始 on-policy distillation（OPD），并研究教师在 student-generated states 上不可靠时的
鲁棒 OPD（ROPD）训练。代码基于 THUNLP 的
[Rethinking On-Policy Distillation](https://github.com/Thinking-Space/Rethinking-OPD) 和 verl v0.7.0，
但仓库内容已收敛到本项目实际使用的训练、可靠性分析和数学评测管道。

## 当前实验设置

- 学生：`DeepSeek-R1-Distill-Qwen-1.5B`；
- 教师：冻结的 `JustRL-DeepSeek-1.5B`；
- 训练数据：`datasets/dapo-math-17k.parquet`；
- 固定评测：AIME24、AIME25、AMC23；
- 训练框架：vendored `verl` 核心包、Ray、FSDP 和 vLLM；
- 大模型、checkpoint、日志和结果位于 `/attached/remote-home1/liufengkai/opd`，不进入 Git。

## 仓库结构

```text
configs/experiments/       OPD、LCB-OPD 训练配置
configs/trustworthy_opd/   教师可靠性和四组分析配置
datasets/                  当前训练集和固定数学测试集
docs/                      实验规范、方法说明和持续改进清单
scripts/                   managed run、tmux、评测和分析入口
verl/verl/                 当前训练依赖的 verl 核心代码及本项目修改
```

`LlamaFactory`、上游 CI/宣传材料、其他算法 recipe、大量示例和无关 benchmark 已从该实验分支移除。
如需查阅原始内容，请使用 `upstream` remote 或 Git 历史，不要重新复制到当前分支。

## 使用环境

```bash
source /remote-home/share/anaconda3/etc/profile.d/conda.sh
conda activate opd
cd /remote-home/liufengkai/projects/OPD

export OPD_ROOT=/remote-home/liufengkai/projects/OPD
export OPD_STORAGE_ROOT=/attached/remote-home1/liufengkai/opd
export OPD_MODEL_DIR=$OPD_STORAGE_ROOT/models
```

环境需要重建时，保留的 verl 安装入口是：

```bash
USE_MEGATRON=0 bash verl/scripts/install_vllm_sglang_mcore.sh
python -m pip install -e ./verl
python -m pip install math-verify
```

## Managed runs

先做快速配置检查：

```bash
python scripts/run_opd_experiment.py \
  configs/experiments/opd_dense_discrete_lcb_opd_probe.yaml \
  --dry-run
```

正式长实验应通过 tmux 启动器运行；启动器会等待并选择满足阈值的空闲 GPU：

```bash
bash scripts/launch_dense_discrete_lcb_comparison_tmux.sh --mode probe
bash scripts/launch_dense_discrete_lcb_comparison_tmux.sh --mode full
```

查看 managed run：

```bash
python scripts/list_opd_runs.py
```

所有运行使用不可变 `RUN_ID`，并保存 resolved config、manifest、精确命令、环境快照、日志、指标和
checkpoint。不要修改已经生成的 run 配置。

## 原始 OPD 复现

有效性复现必须让可训练 actor 参数和 Adam moments 保持 FP32。当前 FSDP 路径中的
`model_dtype=bfloat16` 会同时把参数与优化器矩降为 BF16；在论文的 `1e-6` 学习率下，大部分更新会被
量化为零。managed launcher 默认拒绝这种配置，只有纯管道 smoke test 可以显式豁免。

先单独运行一步 FP32 full-shape probe；启动器默认会在 probe 后停止，不会直接衔接完整训练：

```bash
bash scripts/launch_opd_completion_first_tmux.sh \
  --mode probe \
  --gpu-count 4 \
  --probe-config configs/experiments/opd_fp32_offload_probe.yaml
```

probe 完成后，用下列工具检查论文配置偏差、训练动态、checkpoint/optimizer dtype 以及相对初始模型的
实际参数变化；确认 FP32 更新有效后，才显式改用 `--mode full`：

```bash
python scripts/audit_opd_reproduction.py /absolute/path/to/probe_run \
  --checkpoint-audit
```

如能同时获得 8 张空闲 GPU，可将启动器的 probe/full 配置分别换为
`opd_paper_exact_8gpu_probe.yaml` 和 `opd_paper_exact_8gpu.yaml`，完成无 offload 的论文执行路径核对。

## OPD / ROPD 独立评测

seed 42 的初始学生、OPD step 279 与 ROPD step 279 使用同一批 AIME24、AIME25、AMC23 prompt、
同一解码参数及逐请求随机种子进行配对评测：

```bash
bash scripts/launch_opd_ropd_evaluation_tmux.sh
```

启动器会先检查 143 道题与两个完整 checkpoint，再将 FSDP 权重合并到大容量存储。生成按
`model/dataset/rollout` 原子分片，重启相同 `RUN_DIR` 时只补缺失分片。默认每题 16 次生成，使用
`temperature=0.7`、`top_p=0.95` 和 7168-token response 上限；这个配置用于与既有 seed 42 短预算结果
衔接，不等同于论文的 31,744-token 正式评测。解析失败保留在总体 accuracy 分母中，
并单独报告 parse rate。结果包括逐条输出、分数据集 accuracy/pass@k、按 prompt bootstrap 95% CI、
配对差值和静态 HTML 看板。

论文口径复核和扩展评测使用同一启动器，只替换配置：

```bash
bash scripts/launch_opd_ropd_evaluation_tmux.sh \
  --config configs/experiments/opd_ropd_seed42_paper_aligned_evaluation.yaml \
  --session opd-paper-aligned-eval

bash scripts/launch_opd_ropd_evaluation_tmux.sh \
  --config configs/experiments/opd_ropd_seed42_extended_evaluation.yaml \
  --session opd-extended-eval
```

前者在原论文 143 题上加入教师基线并使用每题 16 次、31,744-token 上限，以计算教师—学生 gap
recovery；后者扩展到 1590 个独立问题、每题 4 次，并在启动前核验冻结的训练集精确重合审计。扩展数据
通过 `scripts/val/prepare_extended_math_eval.py` 准备，审计入口为
`scripts/val/audit_eval_contamination.py`。

```bash
python scripts/val/run_formal_evaluation.py --run-dir /absolute/evaluation/run --status
```

## 更新幅度匹配的 ROPD 消融

原始 OPD、统一缩小到 0.18 倍的 OPD、原始 LCB-ROPD，以及把 LCB-ROPD token reward RMS 恢复到
OPD 水平的 normalized-ROPD 已统一在 seed 43 四臂启动器中。先运行 5 步 probe 检查 reward RMS、
梯度和数值稳定性；不要直接跳过 probe 启动四个完整条件。

```bash
bash scripts/launch_update_matched_comparison_tmux.sh --mode probe
```

## 关键文档

- [实验管理规范](docs/EXPERIMENT_MANAGEMENT.md)
- [持续改进清单](docs/IMPROVEMENT_CHECKLIST.md)
- [OPD / ROPD 证据增强计划](docs/OPD_ROPD_EVIDENCE_PLAN.md)
- [稠密离散邻域与 LCB-OPD](docs/ROBUST_OPD_DENSE_DISCRETE.md)
- [教师可靠性实验设计](docs/TRUSTWORTHY_OPD_PILOT.md)
- [可信 OPD 脚本索引](scripts/trustworthy_opd/README.md)

## 维护原则

- 长实验命令先交由用户审阅，再由用户启动；
- 长实验和持久看板放入 tmux；
- 截断和答案不可解析属于工程有效性问题，不作为学术研究目标；
- OPD 与 ROPD 的正式比较必须使用相同初始 checkpoint、数据顺序、解码参数和评测器；
- 新问题和完成证据持续更新到 `docs/IMPROVEMENT_CHECKLIST.md`。

## 上游与许可

OPD 方法和初始实现来自 THUNLP/Thinking-Space。`verl` 核心代码保留其 Apache-2.0
许可证与 Notice，见 `verl/LICENSE` 和 `verl/Notice.txt`。本分支中的本地修改以 Git 历史为准。
