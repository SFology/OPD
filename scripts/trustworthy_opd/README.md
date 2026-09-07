# Trustworthy OPD 分析脚本

本目录保存教师可靠性、四组 continuation 和 PPL/邻域稳定性分析代码。长任务使用对应的 `launch_*_tmux.sh`
入口；结果写入 `/attached/remote-home1/liufengkai/opd` 下的实验目录，不写回仓库。

## 主流程

- `run_pipeline.sh`：早期可靠性 pilot；
- `launch_paired_content_tmux.sh`：配对内容可靠性实验；
- `launch_four_group_tmux.sh`：四组 continuation 收集；
- `launch_four_group_analysis_tmux.sh`：四组 PPL 与邻域稳定性提取；
- `launch_ppl_decode_control_tmux.sh`：统一师生解码参数后的对照；
- `launch_ppl_recovery_curve_tmux.sh`：教师介入前后 PPL 曲线；
- `launch_ppl_branch_comparison_tmux.sh`：师生分支和 scorer 的完整比较；
- `launch_teacher_degradation_tmux.sh`：教师退化诊断。

## 分层职责

- `run_*.py`：编排、分片、断点续跑和 GPU 调度；
- `*_worker.py`：单 GPU 工作单元；
- `analyze_*.py`：CPU 统计分析和报告；
- `plot_*.py`：图表与 HTML 看板；
- `common.py`、`four_group_common.py`：共享数据结构和工具。

每次运行前应先检查对应 `configs/trustworthy_opd/*.yaml`。不要直接覆盖已经完整落盘的 run；新控制条件
使用新配置和新子目录。可靠性结论必须报告每组/q-point 样本数和 prompt-cluster bootstrap 不确定性。
