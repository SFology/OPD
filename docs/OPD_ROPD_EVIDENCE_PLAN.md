# OPD / ROPD 证据增强计划

最后更新：2026-09-10 UTC

## 1. 当前判断

seed 42 的短预算独立评测中，初始学生总体 average correctness 为 37.15%，原始 OPD 为 38.90%，
配对提升为 `+1.75 pp`，prompt-bootstrap 95% CI 为 `[+0.35,+3.10] pp`。这说明在该评测口径下有一个
小而可检测的正变化，但不能据此判断已经按论文复现了 OPD 的正常收益。

原因是当前评测把 response 限制在 7168 token，初始学生、OPD、ROPD 分别有约 60.45%、58.52%、
58.13% 的输出触及上限。论文的训练 response 上限同样是 7168，但正式评测使用 31,744 token，且以
AIME24、AIME25、AMC23 每题 16 次生成的 avg@16 为主要指标。论文对同一师生组合报告的是最终学生
回收超过 80% 的教师—学生性能差距，而不是一个普遍固定的百分点增益。

因此先以论文口径评估现有 checkpoint，再决定是否重新训练。不能用短预算的 `+1.75 pp` 直接把训练
判定为成功或失败。

## 2. 已识别的混杂与修复

1. **评测长度混杂**：新增 `opd_ropd_seed42_paper_aligned_evaluation.yaml`，使用 31,744-token 上限，
   纳入初始学生、JustRL 教师、OPD 与 ROPD，直接输出 gap recovery 及其配对 bootstrap 区间。
2. **统计不完整**：所有预声明的 pass@k 都输出配对差值、95% CI 和逐题胜/平/负计数；gap recovery
   同样按 prompt bootstrap。随机流按指标命名，增加新图表不会静默改变旧区间。
3. **状态误报**：正式评测进入 merge、analysis 或 complete 时清除过期 model/shard 字段，并有 CPU
   状态转换测试。
4. **答案解析**：训练 verifier 支持完整 boxed 与明确的 Answer/Final answer 标记，但不把推理中的
   任意最后数字当答案；无法解析仍计入主指标分母。
5. **ROPD 更新幅度混杂**：新增 OPD、scaled-OPD、ROPD、normalized-ROPD 四臂。normalized-ROPD
   保留逐 token 相对门控，但按 batch 把 scalar token reward RMS 恢复到 OPD 水平；不使用答案标签。
6. **样本覆盖不足**：扩展评测包含 AIME24、AIME25、AMC23、MATH-500、Minerva 和 Olympiad-Bench，
   共 1590 个独立 prompt、每题 4 次生成。已对 DAPO-Math-17K 做规范化精确重合审计：1590 个评测
   prompt 中精确重合为 0、评测内部重复为 0。该审计不能排除改写或语义污染，因此报告中必须保留这项
   限定。

## 3. 预先声明的阶段与停止门槛

### 阶段 A：论文口径复核现有 seed 42 checkpoint（最优先）

- 143 个 prompt × 16 rollout × 4 个模型，共 9152 次生成；
- 主要量：三个数据集分别的 avg@16、汇总值、OPD 相对初始学生的配对差、gap recovery；
- 工程有效性：parse rate、at-limit rate、完整且可解析比例；
- 判定：只有在教师相对学生确有正 gap 时解释 recovery。若 OPD recovery 与论文“超过 80%”大幅不符，
  才进入阶段 B 的训练数值复核；若一致，则保留现有训练并进入阶段 C/D。

### 阶段 B：有条件的论文训练数值复核

- 先运行 8 GPU、fp32、无参数/优化器 offload 的 1 步 full-shape probe；
- 仅在 probe 成功、阶段 A 仍显示异常低恢复率时，运行完整 fp32 原始 OPD；
- 与当前 bf16/offload 运行比较 reward、grad norm、overlap、最终长预算 avg@16；
- 这一步用于区分“评测口径”与“训练数值/硬件差异”，不是默认必跑项目。

### 阶段 C：ROPD 四臂 5 步更新幅度 probe

- 四臂：OPD、0.18× scaled-OPD、raw ROPD、normalized-ROPD；
- 所有臂都计算同一邻域诊断，避免 measurement overhead 不一致；
- 无 NaN/Inf；scaled-OPD reward RMS 约为 OPD 的 0.18 倍；normalized-ROPD selected reward RMS 与
  OPD 的比值应在 `[0.9,1.1]`；报告 normalization clip/degenerate 比例；
- 若归一化频繁触发 20 倍上限，或梯度仍严重不匹配，不进入完整训练，先校准门控。

### 阶段 D：更新幅度匹配的完整训练与扩展评测

- seed 43 四臂各训练一个 epoch，数据顺序、初始化、rollout 和测量开销保持一致；
- 主要方法对比：`ROPD − scaled-OPD` 与 `normalized-ROPD − OPD`。前者控制全局衰减，后者控制 reward
  RMS；`ROPD − OPD` 只作原始总效应；
- 主要终点：1590-prompt 扩展评测的 average correctness 配对差；次要终点为分数据集效果、训练稳定性、
  effective token mass 与梯度范数；
- 只有两项幅度匹配对比至少一项方向稳定、区间有说服力，并且邻域 risk 能在 held-out 四组数据上区分
  `(T+,S-)` 与 `(T-,S-)`，才把收益解释为“可靠性选择”而非单纯降低学习率。

### 阶段 E：普遍性验证

- 在阶段 D 出现预声明方向后才增加 seed 44，避免先消耗约数百 GPU-hour；
- 对可靠性指标保持 q20/q40/q60/q80 分层，使用 prompt-cluster bootstrap；
- 固定共同 continuation token horizon，分别报告 teacher/student scorer；
- 做表征消融（tail embedding、中间层 prefix/tail mean、last token），并审计邻居文本语义有效性；
- 补充 `(T-,S+)` 必须使用事先固定的计数停止规则；不足时明确标为 underpowered。

## 4. 推荐执行顺序

1. 运行阶段 A；
2. 根据阶段 A 决定是否需要阶段 B；
3. 运行阶段 C，修正任何 reward/gradient 不匹配；
4. 先用现有 seed 42 checkpoint 跑扩展评测，确认数据与 grader；
5. 再运行阶段 D 的 seed 43 完整四臂；
6. 只有效果方向达到预声明门槛才进入阶段 E。

论文依据：<https://arxiv.org/pdf/2604.13016>；上游实现说明：
<https://github.com/Thinking-Space/Rethinking-OPD>。
