# OPD / ROPD 改进清单

最后更新：2026-09-07 UTC

本文件是后续工程修复、实验完善和科学验证的持续清单。发现新的问题时追加条目；完成后保留原条目并将
`[ ]` 改为 `[x]`，同时填写完成日期、commit 或实验目录以及验收结果。不要仅因代码已经写出就标记完成：
涉及实验结论的条目必须有结果文件和验收证据。

状态约定：

- `[ ]`：未完成；
- `[x]`：已经完成并通过所列验收条件；
- `阻塞`：写在条目备注中，并说明恢复条件；
- 优先级从 `P0` 到 `P2` 依次降低。

## 当前最优先

- [ ] **IMP-001（P0）统一训练与评测的数学答案解析。**
  - 问题：DAPO-Math-17k prompt 要求把答案写在 `Answer:` 后，但当前
    `ttrl_math` verifier 只抽取 `\boxed{...}`；正确但未使用 boxed 的训练 rollout 会被记为错误。
  - 改进：实现数据集感知的确定性解析。AIME 至少支持 `\boxed{}` 和明确的最终 `Answer:`，并按
    `0--999` 整数判等；AMC/MATH 保留数值和符号等价判定。不要把推理过程中的任意最后一个数字直接
    当答案。
  - 验收：覆盖 boxed、Answer、前导零、LaTeX、无法解析和错误答案的单元测试；在固定生成样本上人工
    抽查；分别报告 `parse_rate`、总样本 accuracy 和 parse-success conditional accuracy。
  - 相关代码：`verl/verl/utils/reward_score/ttrl_math/`、`scripts/val/eval/grade.py`。

- [ ] **IMP-002（P0）把“教师分布奖励”和“答案正确性”在命名与看板中彻底分开。**
  - 问题：verl 配置中的 `reward_model` 实际承载冻结 teacher policy；`critic/score` 是 OPD token
    reward，而 `critic/true_reward` 才是 verifier correctness，容易误读。
  - 改进：在项目文档、图表和实验摘要中统一标注 `teacher_policy`、`opd_token_reward`、
    `verifier_correctness`；在不破坏上游 verl 接口的前提下增加清晰的别名或解释字段。
  - 验收：新实验的 resolved config、日志摘要和看板不再把教师误称为标量 reward model；三类指标可被
    独立定位。
  - 已完成子项：2026-09-07 将 `true_reward_score` 恢复为可选监控字段；没有 verifier 分数的普通
    OPD batch 不再因指标统计报错，并新增有/无该字段的 CPU 测试。整体命名与看板改进仍未完成。

- [ ] **IMP-003（P0）建立独立、可复现的固定测试集评测流程。**
  - 问题：当前完整训练配置使用 `test_freq=-1`、`val_before_train=false`，训练日志中的
    `critic/true_reward` 只是训练 rollout 正确率，不是 AIME/AMC 泛化性能。
  - 改进：对初始学生、原始 OPD checkpoint、ROPD checkpoint 使用完全相同的生成参数、样本数、随机
    种子和 grader，单独评测 AIME24、AIME25、AMC23；保存逐题逐 rollout 原始输出与评分结果。
  - 验收：每个 checkpoint 都有 manifest、generation JSONL、grading 结果、解析率、pass@1/Avg@N、
    bootstrap 95% CI；报告中明确区分训练监控和 held-out evaluation。

- [ ] **IMP-004（P0）完成严格配对的 OPD / ROPD 训练对比。**
  - 当前状态：完整 OPD 控制臂配置为 `apply_to_training=false`；它会测量 LCB/ROPD，但训练仍使用原始
    OPD reward。不能把该运行当作 ROPD 效果。
  - 改进：控制臂和处理臂仅允许 `apply_to_training` 不同；使用相同初始权重、数据顺序、seed、GPU 数、
    batch、rollout 和测量开销。先确认控制臂完整落盘，再启动处理臂。
  - 验收：两臂均正常完成、无 NaN/Inf、checkpoint 可加载；按 IMP-003 完成统一评测；同时比较训练
    稳定性、奖励分布、长度和耗时。

## ROPD 方法与实现

- [x] **IMP-005（P0）邻居直接计算 anchor action 的精确 log-prob。**
  - 完成内容：不再要求 anchor action 恰好出现在邻居状态的 Top-K 中；对合法邻居直接请求该动作的
    teacher/student log-prob，并记录请求去重、packing 和覆盖率。
  - 证据：`verl/verl/trainer/ppo/robust_opd.py` 中的 exact sampled-action support；当前配置使用
    `action_evaluation=sampled_token_exact`。
  - 完成日期：2026-09-06 之前；后续仍需由 IMP-007 验证邻居语义质量。

- [x] **IMP-006（P0）实现符号保持的绝对幅度 LCB gate。**
  - 完成内容：采用
    `r_LCB = sign(r_OPD) * max(|r_OPD| - lambda * neighborhood_risk, 0)`，避免负 OPD reward 经简单
    下确界后变得更负、更激进。
  - 证据：`verl/verl/trainer/ppo/robust_opd.py`；配置使用 `aggregation=lcb_gate`。
  - 完成日期：2026-09-06 之前；这只表示实现完成，不表示该估计器已经被实验验证有效。

- [ ] **IMP-007（P0）验证并校准当前 LCB trust，避免门控过度塌缩。**
  - 问题：当前控制臂早期日志出现约 89% 的 `zero_trust_fraction`，说明默认
    `lambda=1.0`/risk 标度可能过强；也可能反映邻居质量或风险定义存在问题。
  - 改进：先在不参与训练的固定样本上检查 risk 与教师纠偏正确性的关系，再小规模扫描 `lambda`、风险
    聚合温度和归一化方式。调参不得使用最终测试集标签。
  - 验收：报告 trust/risk 分布、非零有效质量、与 `(T+,S-)` 对 `(T-,S-)` 的区分度及置信区间；明确
    选择阈值的数据来源。

- [ ] **IMP-008（P0）审计稠密离散邻域的语义有效性。**
  - 问题：同 prompt、相近进度和双球约束并不自动保证两个推理状态语义等价；错误邻居会把正常决策变化
    当成教师不稳定。
  - 改进：按 q-point、距离区间、正确性组抽样展示 anchor/neighbor 文本；统计跨 rollout、相对进度差、
    双空间距离、动作变化和 verifier 分组，并进行人工盲审或可复现的语义一致性标注。
  - 验收：给出邻域 precision/接受率及抽样置信区间；任何正式 ROPD 结论都同时报告
    `zero_neighbor_fraction` 和 action coverage。

- [ ] **IMP-009（P1）比较状态表示，不固定在单一 tail-embedding 方案。**
  - 候选：`token_embedding_tail_mean`、中间层 prefix mean、若干层 tail mean、最后 token hidden state，
    以及经过充分论证的新增表示。
  - 验收：在完全相同的 frozen anchor/support 上做消融；比较邻居重合率、语义有效性、教师可靠性区分度、
    显存和时间，不以训练最终性能单独选择表示。

- [ ] **IMP-010（P1）评估更稠密但仍高效的跨 batch / 跨 step support memory bank。**
  - 前提：IMP-008 证明当前邻域定义至少有可接受的语义 precision。
  - 改进：按 prompt/问题维护带版本和过期策略的状态缓存，防止过旧策略状态混入；先离线重放，再决定
    是否接入在线训练。
  - 验收：相对当前同 batch support 明显提高有效邻居覆盖，同时邻域 precision 不下降，额外耗时和存储
    有清晰预算。

- [ ] **IMP-011（P2）仅在离散方法证据充分后研究连续邻域近似。**
  - 备选：soft-token/embedding 对抗扰动、奖励对状态表示的一阶线性化与梯度范数上界、共享语义编码器。
  - 验收前置条件：明确教师和学生不同表示空间中的共同语义约束，并验证局部近似误差；在此之前不作为
    主实验方法。

## 可靠性指标与科学验证

- [ ] **IMP-012（P1）在统一解码的四组数据上完成邻域稳定性验证。**
  - 主对比：`(T+,S-)` 对 `(T-,S-)`；`(T+,S+)` 用于检验是否只是一般轨迹规律；`(T-,S+)`
    用于研究有害教师，但在扩样前仅作描述。
  - 控制：q20/q40/q60/q80 分开，固定共同 continuation token horizon，教师和学生 scorer 分开。
  - 验收：prompt-cluster bootstrap 95% CI、效应量、AUROC/AUPRC、校准曲线和每组样本数齐全；不得仅以
    policy-origin 分离证明教师可靠性。

- [ ] **IMP-013（P1）去除模型 self-preference 后再评估可靠性信号。**
  - 问题：教师偏好教师分支、学生偏好学生分支；raw PPL/likelihood 很大程度反映文本来自哪个策略，而
    不是内容是否正确。
  - 改进：构造去除 scorer/branch origin 主效应的 calibrated residual，并与 raw PPL、局部稳定性对比。
  - 验收：校准只使用训练/开发划分；在 held-out prompt 上仍能区分成功与失败纠偏，并报告不确定性。

- [ ] **IMP-014（P1）补充 `(T-,S+)` 有害介入样本。**
  - 当前限制：统一解码后的有效样本只有约 20 个，无法支持按四个 q-point 的稳定推断。
  - 验收：事先写明计数停止规则；达到每个 q-point 的最低有效样本数或达到预设算力上限；未达到时如实
    标注 underpowered，不反复观察结果后改变停止规则。

- [ ] **IMP-015（P1）量化无效输出造成的选择偏差，但不把截断作为学术问题。**
  - 改进：修复长度/格式工程问题后，只在可解析、非截断样本上研究教师可靠性；同时报告各组/q-point
    纳入率，并对 late-q 结论做敏感性分析。
  - 验收：有效性过滤规则在看结果前固定；解析失败、长度上限和其他排除原因分开统计。

- [ ] **IMP-016（P1）用多 seed 和预先声明的主要终点支持普遍性结论。**
  - 改进：至少复现实验于多个训练 seed；主要终点优先为 AIME24/AIME25/AMC23 held-out correctness、
    训练稳定性和可靠性指标区分度，避免在大量指标中事后挑选。
  - 验收：报告 seed 间方差、配对差值 CI、数据集分层结果；清楚区分探索性与验证性分析。

## 实验管理与维护

- [x] **IMP-017（P0）建立 managed-run 目录与可追溯产物。**
  - 完成内容：resolved config、manifest、command、status、环境快照、日志、checkpoint、SwanLab 和独立
    evaluation 目录均按 RUN_ID 管理。
  - 证据：`docs/EXPERIMENT_MANAGEMENT.md` 和现有 `${OPD_STORAGE_ROOT}/experiments/<RUN_ID>`。

- [x] **IMP-018（P0）整理并提交当前尚未纳入版本控制的可信 OPD 分析代码。**
  - 问题：仓库目前有多项 modified/untracked 的 `configs/trustworthy_opd/` 和
    `scripts/trustworthy_opd/` 文件；正式实验若基于脏工作树，复现和交接会变困难。
  - 改进：逐项确认归属，删除或归档真正无用的临时文件，为保留代码补最小运行说明和快速测试；不要
    覆盖用户已有改动。
  - 验收：相关代码通过编译/短测试并形成范围清晰的 commit；正式运行的 manifest 指向该 commit，必要
    的未提交 diff 为零。
  - 完成情况：2026-09-07 将可信 OPD 配置、编排、worker、分析和绘图脚本纳入版本控制；增加脚本与
    配置索引；移除 LlamaFactory、上游示例/recipe/CI/文档、无关数据和旧 hard-min 入口。本条所在整理
    commit 通过 350 个 Python 文件 AST 检查、全部 Shell 语法检查、24 个 YAML 解析检查、两套 LCB
    probe dry-run，以及 44 个 OPD/ROPD CPU 测试。

- [ ] **IMP-019（P1）让本清单进入每次正式实验的收尾流程。**
  - 改进：实验结束后检查是否完成某个条目、是否暴露新问题；更新日期、状态、证据目录和新的待办。
  - 验收：下一次正式实验报告或 handoff 明确引用本文件，已完成事项具有 commit/run/result 证据。

- [ ] **IMP-020（P2）消除数学 grader 中的 Python 非法转义告警。**
  - 发现日期：2026-09-07。
  - 问题：`scripts/val/eval/utils.py` 和 `verl/utils/reward_score/ttrl_math/` 中的部分正则表达式与
    LaTeX 字符串没有使用 raw string，Python 3.12 静态解析会产生 `SyntaxWarning`。当前语义通常仍可
    运行，但未来 Python 版本可能收紧处理。
  - 验收：改写后在 `PYTHONWARNINGS=error` 下导入相关 grader 无告警；现有及 IMP-001 新增答案解析
    测试全部通过。

## 新增条目模板

复制以下模板追加到对应章节；编号递增，不复用已删除或已完成的编号。

```markdown
- [ ] **IMP-XXX（P0/P1/P2）简短标题。**
  - 发现日期：YYYY-MM-DD
  - 问题：为什么需要改进。
  - 改进：准备怎么做。
  - 验收：满足哪些可检查条件才允许勾选。
  - 证据：commit、run 目录、结果文件或测试命令；未完成时写“待补充”。
```
