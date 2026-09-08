# 稠密离散邻域与 LCB-OPD

## 目标与对照原则

原始 sampled-action OPD 奖励为：

\[
r_{\mathrm{OPD}}(s,v)=\log \pi_T(v\mid s)-\log \pi_{\theta_{old}}(v\mid s).
\]

直接对该有符号奖励取邻域下确界会使负奖励变得更负，违背“教师不稳定时少学一点”的目标。当前实现
因此采用符号保持的幅度收缩：

\[
d(s,v)=\operatorname{Agg}_{s'\in\widehat{\mathcal B}_\rho(s)}
\left|r_{\mathrm{OPD}}(s',v)-r_{\mathrm{OPD}}(s,v)\right|,
\]

\[
r_{\mathrm{LCB}}(s,v)=\operatorname{sign}(r_{\mathrm{OPD}}(s,v))
\left[|r_{\mathrm{OPD}}(s,v)|-\lambda d(s,v)\right]_+.
\]

定义 `trust=|r_LCB|/|r_OPD|`，再用 trust 缩放原始 Top-K OPD token reward。无合法邻居时
`trust=1`，不会因为支撑缺失而凭空惩罚。代码还断言 LCB 后的奖励绝对值不得增大。

OPD 控制臂和 LCB-OPD 处理臂都会计算并落盘全部邻域诊断；两臂唯一的训练目标差异是
`apply_to_training`，从而匹配额外测量开销。

## 稠密离散支撑集

支撑集只使用同一 prompt 在当前 on-policy batch 中的其他 student rollouts，不混合不同题目。对每个
anchor state `(rollout_i, token_t)`：

1. 按另一条 rollout 的归一化进度定位中心；
2. 在中心附近按 `candidate_stride` 和 `candidate_offsets` 稠密取候选；
3. 只保留归一化进度差不超过 `progress_window` 的状态；
4. 分别计算学生空间和教师空间的余弦距离；
5. 取两个空间各自最近 `radius_quantile` 的交集；
6. 再施加教师和学生绝对距离上限；
7. 从双球交集中取联合距离最小的 `neighbor_k` 个邻居。

该方法仍是经验分布上的有限支撑近似，不等于连续语义球。邻域的语义有效性必须单独审计，见
`IMPROVEMENT_CHECKLIST.md` 中 IMP-008。

## 状态表示

当前在线实现分别使用两套模型自己的 input embedding。状态 `s_t=(x,y_{<t})` 不包含当前动作 token；
默认对末尾 16 个有效 token 的 embedding 做均值池化和 L2 归一化。

教师和学生 embedding 不跨模型直接比较。同一对状态分别得到 teacher distance 和 student distance，
再通过双球交集决定是否合法，因此不要求两者共享 embedding space。为降低 CPU 和存储开销，embedding
表使用固定随机种子的 32 维 CountSketch 投影，缓存位于：

```text
/attached/remote-home1/liufengkai/opd/cache/robust_opd_embeddings
```

状态表示尚未被证明最优，必须按 IMP-009 与 contextual hidden-state 方案做受控消融。

## Anchor action 的精确邻居评分

旧实现只能在邻居 Top-K 也包含 anchor action `v` 时比较奖励，导致大量动作质量不可比较。当前实现为
每个合法邻居直接请求：

```text
log pi_T(v | s')
log pi_S(v | s')
```

请求按目标 trajectory 打包，并对相同 `(position, action_id)` 去重，不再把 Top-K 未命中误当成零概率。
风险由邻居 reward 与 anchor reward 的绝对差得到，支持 `max` 或 softmax-weighted 聚合。

关键监控包括：

- `ropd/zero_neighbor_fraction`：没有合法邻居的状态比例；
- `ropd/action_neighbor_weighted_coverage`：按 OPD 权重计算的精确动作覆盖；
- `ropd/exact_request_dedup_fraction`：稀疏动作请求去重比例；
- `ropd/neighborhood_risk_*`：邻域奖励偏差；
- `ropd/trust_*`、`ropd/zero_trust_fraction`：实际门控强度；
- `ropd/opd_token_reward_*`、`ropd/ropd_token_reward_*`：缩放前后奖励分布；
- `ropd/applied_to_training`：本臂是否真的用 LCB reward 更新学生。

每个 managed run 还会写入：

```text
RUN_DIR/metrics/ropd_step_metrics.jsonl
RUN_DIR/metrics/ropd_reward_samples.jsonl
```

样本文件保留 anchor action、精确师生 log-prob、risk、trust、邻居位置和双空间距离，供异常审计。

## 当前配置与启动入口

- `configs/experiments/opd_dense_discrete_lcb_base.yaml`：两臂共享配置；
- `configs/experiments/opd_dense_discrete_lcb_opd_probe.yaml`：一步 OPD 控制探针；
- `configs/experiments/opd_dense_discrete_lcb_treatment_probe.yaml`：一步 LCB-OPD 处理探针；
- `configs/experiments/opd_dense_discrete_lcb_opd.yaml`：完整 OPD 控制臂；
- `configs/experiments/opd_dense_discrete_lcb_treatment.yaml`：完整 LCB-OPD 处理臂；
- `scripts/launch_dense_discrete_lcb_comparison_tmux.sh`：等待并自动选择空闲 GPU，顺序运行配对实验。

旧的 hard-min 配置和启动器已经移除；如需追溯，可从 Git 历史恢复，但不得与当前 LCB 结果混为一谈。

## 启动完整实验前的验收

1. 两个 probe 都完成一步并且没有 NaN/Inf；
2. 任意 token 的 LCB reward 绝对值均不大于原 OPD reward；
3. 合法邻居与精确动作覆盖率不是接近零；
4. `zero_trust_fraction` 已解释或校准，不能在默认门控近乎全零时直接进行正式比较；
5. `compute_robust_opd` 耗时和 sparse request 数量在预算内；
6. 抽查样本的 UID、状态、动作、邻居位置和双空间距离合理；
7. 两臂除 `apply_to_training` 外的 resolved config 完全一致。

## 暂不实现的方向

- 跨 batch / 跨 step、按 prompt 建立的 policy-versioned memory bank；
- soft-token 或 embedding 空间连续对抗扰动；
- 奖励关于状态表示的一阶线性化和梯度范数上界；
- 单独训练共享语义编码器或用任务级文本编辑生成邻域。

这些方向保留在持续改进清单中；在离散邻域语义有效性尚未验证前，不接入正式训练。
