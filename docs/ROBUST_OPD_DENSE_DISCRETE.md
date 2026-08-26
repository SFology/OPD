# 稠密离散轨迹邻域 ROPD

## 目标与对照原则

本实现比较原始 OPD 奖励

\[
r_{\mathrm{OPD}}(s,v)=\log \pi_T(v\mid s)-\log \pi_{\theta_{old}}(v\mid s)
\]

与有限支撑集上的悲观奖励

\[
r_{\mathrm{ROPD}}(s,v)=\min_{s'\in\widehat{\mathcal B}_\rho(s)}
r_{\mathrm{OPD}}(s',v).
\]

OPD 对照组和 ROPD 处理组都会计算并落盘两种奖励及邻域诊断；两组唯一的训练目标差异是
`apply_to_training`。这可避免把额外测量开销或邻域覆盖差异混入方法对比。

## 已实现的稠密离散支撑集

支撑集只使用同一 prompt 在当前 on-policy batch 中的其他 student rollouts，不把不同题目的状态混在
一起。对每个锚点状态 `(rollout_i, token_t)`：

1. 按另一条 rollout 的归一化进度对齐中心位置；
2. 在中心附近以 `candidate_stride` 和 `candidate_offsets` 稠密取候选状态；
3. 仅保留归一化进度差不超过 `progress_window` 的状态；
4. 分别计算学生空间与教师空间的余弦距离；
5. 取两个空间各自最近的 `radius_quantile`，再取交集（双球约束）；
6. 施加两个绝对距离上限，避免“候选都很远时仍保留排名第一”的问题；
7. 从交集中取联合距离最小的 `neighbor_k` 个邻居并执行 hard minimum。

没有合法邻居时使用 `ROPD = OPD`，不会为了覆盖率强行加入远状态。

## 状态表示

第一版在线实现采用两套模型各自的 input embedding 表。状态 `s_t` 严格只包含 prompt 与
`y_{<t}`，不包含当前动作 token `y_t`。默认对末尾 16 个有效 token 的 embedding 做均值池化，
然后 L2 归一化。

教师和学生 embedding 不在同一个坐标系中，因此不会直接比较二者向量。相反，同一对状态分别得到
教师距离和学生距离，并用双球交集决定是否合法。这样不要求共享输入嵌入器。

为了降低 CPU 与存储开销，embedding 表使用固定随机种子的 32 维 CountSketch 投影。投影保持内积的
无偏估计，且构建成本相对 embedding 表大小为线性。缓存位于：

```text
/attached/remote-home1/liufengkai/opd/cache/robust_opd_embeddings
```

两个实验臂和后续运行会复用缓存。

## 动作支持与精确奖励

原仓库以学生 Top-16 动作近似反向 KL。对锚点 Top-16 中的动作 `v`，只有当邻居状态的学生 Top-16
也包含 `v` 时，该邻居才提供现成且精确对应的 teacher/student log-prob。未匹配动作不会被当作零概率
或伪造极端奖励，而是跳过，并记录 action coverage。

因此需要同时看两个量：

- `ropd/zero_neighbor_fraction`：状态层面没有合法邻居的比例；
- `ropd/action_neighbor_weighted_coverage`：按锚点 OPD 权重统计，动作能在邻居 Top-16 中找到的质量。

覆盖率过低时，不应直接解释 ROPD 效果；应优先调邻域半径、候选密度，必要时再增大 Top-K。

## 输出

每个 managed run 除常规日志外，还会写入：

```text
RUN_DIR/metrics/ropd_step_metrics.jsonl
RUN_DIR/metrics/ropd_reward_samples.jsonl
```

步级文件包含 OPD/ROPD token reward、悲观惩罚的均值和分位数、邻居数、覆盖率、实际改变比例以及
`apply_to_training`。样本文件记录锚点动作、原奖励、鲁棒奖励、最坏邻居位置和双空间距离，便于定位
异常邻居。

## 配置与比较实验

- `configs/experiments/opd_dense_discrete_opd_probe.yaml`：一步 OPD 对照探针；
- `configs/experiments/opd_dense_discrete_ropd_probe.yaml`：一步 ROPD 处理探针；
- `configs/experiments/opd_dense_discrete_opd.yaml`：完整 OPD 对照；
- `configs/experiments/opd_dense_discrete_ropd.yaml`：完整 ROPD 处理；
- `scripts/launch_dense_discrete_ropd_comparison_tmux.sh`：自动选择稳定空闲 GPU，顺序运行两组。

先跑 probe，确认以下条件再启动完整对比：

1. 两组均完成一步且没有 NaN/Inf；
2. ROPD raw reward 从未大于对应 OPD raw reward；
3. 合法邻居与动作覆盖率不是接近零；
4. `compute_robust_opd` 耗时相对整步可接受；
5. 抽查 JSONL 中的邻居属于同一 UID，且状态/动作位置合理。

## 当前边界与暂不实现的方向

这一版比小型同 batch 状态集合更稠密，但仍是经验分布上的有限支撑近似，不等于对真正连续球求
精确下确界。下列方向仅保留为后续方案，本轮不实现：

- 中间层最后 token、若干层拼接或 attention-weighted contextual hidden state；
- 跨 batch / 跨 step 的分 prompt memory bank；
- 在 soft-token 或 embedding 空间中进行连续对抗扰动；
- 对奖励关于状态表示做一阶线性化，并以梯度范数近似连续球最坏情况；
- 单独训练共享语义编码器或用任务级文本编辑生成邻域。

其中连续 soft-token 方案需要明确教师/学生不同 embedding 空间之间的语义约束；线性化方案则需要
额外反向传播并验证局部近似误差。它们不适合作为当前首个受控对比的默认实现。
