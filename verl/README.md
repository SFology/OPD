# Vendored verl core for OPD / ROPD

该目录保留本项目运行 OPD/ROPD 所需的 verl v0.7.0 核心包、安装元数据和环境安装脚本。上游的大量
文档、CI、Docker、示例、其他算法 recipe 和测试已从当前实验分支移除，以免与本项目入口混淆。

当前项目修改主要位于：

- `verl/trainer/ppo/`：token-level OPD、稠密离散邻域和 LCB-OPD；
- `verl/workers/`：teacher log-prob、rollout 与 reward 数据流；
- `verl/utils/reward_score/ttrl_math/`：数学答案解析和正确性监控；
- `verl/trainer/config/`：Hydra 运行配置。

从仓库根目录安装：

```bash
python -m pip install -e ./verl
```

项目实验请使用根目录的 `scripts/run_opd_experiment.py` 和 `configs/`，不要直接在本目录添加新的运行
入口。完整的上游文档与示例见 <https://github.com/volcengine/verl>；许可证和 Notice 保留在本目录。
