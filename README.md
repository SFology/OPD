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

## 关键文档

- [实验管理规范](docs/EXPERIMENT_MANAGEMENT.md)
- [持续改进清单](docs/IMPROVEMENT_CHECKLIST.md)
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
