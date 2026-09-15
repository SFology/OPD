# OPD / ROPD 改进清单

最后更新：2026-09-11 UTC

本文件是后续工程修复、实验完善和科学验证的持续清单。发现新的问题时追加条目；完成后保留原条目并将
`[ ]` 改为 `[x]`，同时填写完成日期、commit 或实验目录以及验收结果。不要仅因代码已经写出就标记完成：
涉及实验结论的条目必须有结果文件和验收证据。

维护约定：在后续会话中，只要分析或复盘报告提出了新的、可执行的工程或科学改进项，就在同一轮工作中
自动追加到本清单或补充已有条目，无需等待再次提醒。完成改进后同步勾选并填写证据；纯猜想在尚未形成
可执行改进和验收标准前不登记。

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
  - 进展：2026-09-10 已让 `ttrl_math` 同时支持完整 `\boxed{...}`、行级 `Answer:`、`Final answer:`
    和 `The final answer is ...`，同时明确禁止把未标注的最后一个数字作为答案；前导零与整数/浮点表示
    沿用符号等价判定。新增 9 个 CPU 测试覆盖正确、错误、前导零、LaTeX、未标注推理数字和未闭合
    box，全部通过。仍需在正式生成完成后做固定样本人工抽查，故本条暂不勾选。

- [ ] **IMP-002（P0）把“教师分布奖励”和“答案正确性”在命名与看板中彻底分开。**
  - 问题：verl 配置中的 `reward_model` 实际承载冻结 teacher policy；`critic/score` 是 OPD token
    reward，而 `critic/true_reward` 才是 verifier correctness，容易误读。
  - 改进：在项目文档、图表和实验摘要中统一标注 `teacher_policy`、`opd_token_reward`、
    `verifier_correctness`；在不破坏上游 verl 接口的前提下增加清晰的别名或解释字段。
  - 验收：新实验的 resolved config、日志摘要和看板不再把教师误称为标量 reward model；三类指标可被
    独立定位。
  - 已完成子项：2026-09-07 将 `true_reward_score` 恢复为可选监控字段；没有 verifier 分数的普通
    OPD batch 不再因指标统计报错，并新增有/无该字段的 CPU 测试。整体命名与看板改进仍未完成。

- [x] **IMP-003（P0）建立独立、可复现的固定测试集评测流程。**
  - 问题：当前完整训练配置使用 `test_freq=-1`、`val_before_train=false`，训练日志中的
    `critic/true_reward` 只是训练 rollout 正确率，不是 AIME/AMC 泛化性能。
  - 改进：对初始学生、原始 OPD checkpoint、ROPD checkpoint 使用完全相同的生成参数、样本数、随机
    种子和 grader，单独评测 AIME24、AIME25、AMC23；保存逐题逐 rollout 原始输出与评分结果。
  - 验收：每个 checkpoint 都有 manifest、generation JSONL、grading 结果、解析率、pass@1/Avg@N、
    bootstrap 95% CI；报告中明确区分训练监控和 held-out evaluation。
  - 完成证据：2026-09-10 的正式评测
    `20260910_opd_ropd_seed42_formal_evaluation` 已以退出码 0 完成；初始学生、OPD step 279、ROPD
    step 279 各完成 48/48 shards、2288/2288 条生成，共 6864 条且配对 seed 不匹配数为 0。目录中保留
    manifest、逐条 generation/grading、数据集分层 summary、配对 prompt-bootstrap 95% CI、SVG/HTML
    看板，并已回写两个训练 run 的 evaluation 指针。训练监控与 held-out evaluation 在报告中分开。

- [ ] **IMP-004（P0）完成严格配对且更新有效的 OPD / ROPD 训练对比。**
  - 当前状态：完整 OPD 控制臂配置为 `apply_to_training=false`；它会测量 LCB/ROPD，但训练仍使用原始
    OPD reward。不能把该运行当作 ROPD 效果。
  - 改进：控制臂和处理臂仅允许 `apply_to_training` 不同；使用相同初始权重、数据顺序、seed、GPU 数、
    batch、rollout 和测量开销。先确认控制臂完整落盘，再启动处理臂。
  - 验收：两臂均正常完成、无 NaN/Inf、checkpoint 可加载；按 IMP-003 完成统一评测；同时比较训练
    稳定性、奖励分布、长度和耗时。
  - 历史进展：原始 OPD 控制臂
    `20260906_045836_opd_dense_discrete_lcb_opd_seed42_c200eae` 已于 2026-09-07 完成 279 步；ROPD
    处理臂 `20260907_164248_opd_dense_discrete_lcb_treatment_seed42_693bda4` 已于 2026-09-09 完成
    279 步，退出码为 0；两个 run 的最终 FSDP model/optimizer rank 0/1 shard 和 tokenizer/config 均存在。
    独立评测也已完成：固定预算总体 accuracy 为 OPD 38.90%、ROPD 38.94%，配对差
    `ROPD-OPD=+0.04 pp`、95% CI `[-1.49,+1.53] pp`，因此本轮完成了严格比较，但没有证据表明默认
    ROPD 优于 OPD。2026-09-11 的 IMP-030/032 审计进一步证明两臂均使用 BF16 actor/Adam moments，
    `1e-6` 更新大面积被量化吞掉。因此产物配对和评测流程虽然完成，但这不是“更新有效”的方法效果对比，
    本条重新打开；需在 FP32 原始 OPD 复现通过后重跑至少一个严格配对的 FP32 ROPD 条件。

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
  - 完整运行证据：seed 42 处理臂 279 步的平均 `zero_trust_fraction=86.46%`、平均
    `trust_mean=0.115`，最终一步分别为 90.13% 和 0.080。相对 OPD 控制臂，处理臂平均 actor
    `grad_norm` 从 1.812 降至 0.323（约 -82%），训练 rollout verifier correctness 仅从 19.87%
    描述性变为 20.07%。这支持“默认门控主要在压低学习信号”的担忧，但尚未证明门控与教师可靠性相关。
  - FP32 探针证据（2026-09-15）：已完成的 seed-43 OPD 诊断臂 5 步平均邻居覆盖率为 92.28%、
    `zero_neighbor_fraction=7.72%`、全体 `trust_mean=8.66%`、`zero_trust_fraction=89.46%`。由于实现会把
    无邻居状态强制回退为 `trust=1`，换算后有邻居状态的平均 trust 仅 1.02%，其中 96.95% 被清零。
    640 个均匀抽样状态的 `|anchor reward|` 中位数为 0.0773，而 risk 中位数为 3.6205、
    `risk/|reward|` 中位数为 36.40；默认 `lambda=1` 的标度明显过强。后续必须在一次前向中记录预注册
    lambda 网格的反事实 trust/RMS，并按“有邻居/无邻居”分解 reward RMS、梯度贡献和参数更新；无邻居
    回退应作为独立消融，不能把“缺少证据”默认解释为“教师可靠”。在完成这些诊断及 IMP-008/012 的
    语义与正确性校准前，不启动默认 `lambda=1` 的完整 ROPD 训练。
  - 下一实验准备（2026-09-15）：新增独立 seed-44 FP32 OPD 诊断配置，在同一次已完成的邻域前向上对
    预注册 `lambda=[0,0.003,0.01,0.03,0.1,0.3,1]` 计算全量反事实 trust、零信任率、selected RMS、
    绝对 reward mass 保留率及无邻居回退占比；这些统计不改变 actor update，也不读取 correctness 标签。
    启动器会先完成四臂 checkpoint 参数更新审计，再运行该 5-step 诊断并自动汇总结果。
  - 诊断结果（2026-09-15）：seed-44 FP32 OPD 轨迹已完成 5/5 step，编排退出码 0。仅统计有邻居状态时，
    `lambda=0/0.003/0.01/0.03/0.1/0.3/1` 的平均 trust 分别为
    `95.50%/57.19%/46.68%/34.12%/18.69%/7.08%/1.02%`，零 trust 比例分别为
    `4.50%/35.42%/43.16%/53.36%/68.93%/84.71%/96.95%`；绝对 reward mass 保留率分别为
    `100.00%/95.92%/89.91%/78.60%/57.50%/33.80%/15.59%`。这再次排除默认 `lambda=1`，并把
    `0.003--0.03` 确定为后续 correctness 校准应优先考察的量级，但本诊断未读取正确性标签，不能据此
    选择最终 lambda 或声称门控能识别教师可靠性。无邻居回退在 `lambda=1` 的 selected absolute reward
    中占 79.66%，因此后续必须将回退策略作为显式消融。

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
  - 下一实验安排（2026-09-15）：先在现有统一解码冻结数据上做 discovery calibration。此前与旧邻域
    特征相交的保守预估是 610 个状态、56 个 prompt；新管道不依赖旧标签筛选，实际完整计数见下一项。
    按 prompt 分组做交叉验证并用 prompt-cluster bootstrap 报告不确定性；不得随机拆分相关状态。主实现
    需重建与在线训练一致的稠密离散 support（同 prompt、
    跨 rollout、进度窗、双空间约束、anchor action 精确 log-prob），比较当前
    `token_embedding_tail_mean` 与预注册 contextual 表示，评估 raw risk、`risk/|r_OPD|` 和
    `lambda=0.003/0.01/0.03` 的 trust。表示和 lambda 选定后，再在不重叠的新 prompt 块上作一次冻结的
    confirmatory test；当前批次不作为最终确认集。
  - 实现进展（2026-09-15）：新增可恢复的离线校准管道、自动空闲 GPU 分片、prompt-fold、精确
    anchor-action 请求去重、prompt-cluster bootstrap、AUROC/AUPRC/校准曲线、表示/支撑密度消融及 HTML
    邻居文本审计。`--prepare-only` 已验证冻结清单并生成 run
    `20260915_110808_lcb_reliability_discovery`：512 条轨迹、1,331 个统一解码有效 anchor、25,815 个唯一
    状态点和 121,090 条候选边。四个 q 的主对比 `(T+,S-):(T-,S-)` 分别为
    `143:94/102:95/64:80/44:55`。这仍是发现集，模型特征/精确评分和最终统计尚待长任务运行，因此本条
    不勾选。

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
  - 当前证据：7168 是完整 response（推理过程加最终答案）的生成上限，不是最终答案字段长度。当前完整
    控制臂平均 response 长度约 5687 token、约 53.9% 轨迹触及上限；完整处理臂均值约 5624 token、
    约 52.5% 轨迹触及上限，说明该上限是显著的工程有效性约束。
  - 改进：原始 OPD 复现保留 7168 以保持口径；用于方法结论的新 OPD/ROPD 对照应先通过 IMP-022
    标定更长预算，并让两臂使用完全相同的长度配置。修复长度/格式工程问题后，只在可解析、非截断样本上
    研究教师可靠性；同时报告各组/q-point 纳入率，并对 late-q 结论做敏感性分析。
  - 验收：有效性过滤规则在看结果前固定；解析失败、长度上限和其他排除原因分开统计。
  - 进展：2026-09-10 的正式评测分析已预先固定两套口径：固定 7168-token 预算的总体 accuracy 为主
    指标；另外分别报告 parse rate、at-limit rate、可解析且未触及上限的比例及其条件正确率。条件结果
    只作为工程有效性敏感性检查，不把截断包装为学术发现。完整评测中初始/OPD/ROPD 的 parse rate
    分别为 41.78%/43.66%/44.14%，at-limit rate 为 60.45%/58.52%/58.13%，所有未触及长度上限的输出
    均可解析；可解析条件正确率分别为 88.91%/89.09%/88.22%。这说明低解析率主要来自 7168-token
    工程上限，而非当前 parser 漏判；更长预算与有效样本敏感性分析仍未完成，故本条不勾选。

- [ ] **IMP-016（P1）用多 seed 和预先声明的主要终点支持普遍性结论。**
  - 改进：以严格配对的 seed 42 为首个完整对照；主要效应方向稳定后，再补 seed 43、44，形成至少三个
    训练 seed。每个 seed 的 OPD/ROPD 必须共享初始权重、数据顺序、解码参数和评测随机数方案。主要终点
    优先为扩展后的 held-out correctness、训练稳定性和可靠性指标区分度，避免在大量指标中事后挑选；
    不以简单增加 epoch 代替独立 seed。
  - 资源预算：当前 2×RTX 6000 Ada、7168 token、完整 DAPO-Math-17k 的单条件实测约 35.7 小时；在
    seed 42 已完成配对的前提下，补两个 seed 的四个条件约需 143 小时双卡时间。实际启动前用 IMP-022
    的短程标定更新预算。
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

- [x] **IMP-019（P1）让本清单进入每次正式实验的收尾流程。**
  - 改进：实验结束后检查是否完成某个条目、是否暴露新问题；更新日期、状态、证据目录和新的待办。
    按 2026-09-08 的维护约定，任何分析报告中形成的可执行改进应在报告后自动登记，不再等待用户再次
    发出“写入待办”的指令。
  - 验收：下一次正式实验报告或 handoff 明确引用本文件，已完成事项具有 commit/run/result 证据。
  - 完成证据：2026-09-10 在 seed 42 配对训练完成后的状态检查中，自动更新 IMP-003/004/007 的状态和
    运行证据，并新增 IMP-024；OPD 交接 skill 已要求后续报告继续执行同一规则。

- [ ] **IMP-020（P2）消除数学 grader 中的 Python 非法转义告警。**
  - 发现日期：2026-09-07。
  - 问题：`scripts/val/eval/utils.py` 和 `verl/utils/reward_score/ttrl_math/` 中的部分正则表达式与
    LaTeX 字符串没有使用 raw string，Python 3.12 静态解析会产生 `SyntaxWarning`。当前语义通常仍可
    运行，但未来 Python 版本可能收紧处理。
  - 验收：改写后在 `PYTHONWARNINGS=error` 下导入相关 grader 无告警；现有及 IMP-001 新增答案解析
    测试全部通过。

- [x] **IMP-026（P0）建立本轮 OPD/ROPD 的受管、可恢复独立评测启动器。**
  - 发现日期：2026-09-10。
  - 问题：现有 `scripts/val/eval/gen_vllm.py` 硬编码 8 张 GPU、旧模型路径和 31744-token 上限，不能
    直接用于 seed 42 的 7168-token OPD/ROPD 配对评测；两个 FSDP checkpoint 还需先合并为
    HuggingFace 格式。
  - 改进：自动选择空闲 GPU，依次评测初始模型、OPD step 279 和 ROPD step 279；冻结 AIME24、
    AIME25、AMC23 的 143 题、每题 16 次生成、temperature 0.7、top-p 0.95、7168-token 上限及随机数
    方案。checkpoint 合并、逐 shard 生成、数据集感知评分、bootstrap 汇总均需可恢复，并写入各训练
    run 的 `evaluation/` 子目录。
  - 资源预算：三个模型共 6864 条生成；依据本轮平均 response 长度约 5600 token 和训练实测吞吐，
    2 GPU 初步估计约 7--10 小时墙钟时间，两个训练后 checkpoint 约 4.5--7 小时。启动前用固定的
    10 题×1 rollout 短测校准吞吐并更新 ETA。
  - 验收：dry-run 验证模型、题目数、生成数和参数；短测与完整任务均无硬编码 GPU；断点重启不会覆盖
    已完成 shard；输出逐题结果、解析率、分层 accuracy/pass@k、prompt-cluster bootstrap 95% CI，且
    明确区分初始、OPD、ROPD。
  - 进展证据：2026-09-10 已加入配置、FSDP 合并、自动空闲 GPU 选择、原子 shard、损坏 shard 隔离、
    三次失败重试、状态文件、逐请求配对 seed、确定性 grader、prompt-bootstrap、SVG/HTML 看板和 tmux
    启动器。静态编译、Shell 语法、Ruff、13 个 CPU 测试和 prepare-only 均通过；prepare-only 实测
    `143 prompts / 6864 generations / 48 shards per model`，最长 prompt 为 413 token。随后完整任务自动
    选择 GPU 2/3，于 2026-09-10 11:42--13:44 UTC 在约 2 小时 2 分钟内以退出码 0 完成；三模型均为
    48/48 shards、2288/2288 条，分析结果与看板齐全，断点/原子 shard 机制未发现损坏产物。
    2026-09-14 又为 FP32 复现评测增加冻结 generation 复用：只有生成配置、完整 prompt manifest、
    模型源路径、全部 shard 和逐请求 seed 均严格匹配时，才把既有控制输出原子导入新 run；真实数据
    preflight 已验证可复用 initial/teacher/旧 BF16 OPD 共 144 shards、6864 条，仅需为新 FP32 OPD
    生成 48 shards、2288 条，避免无信息的重复计算。

## 规模扩展与算力标定

- [ ] **IMP-021（P0）把 held-out 评测扩展到足够多的独立问题。**
  - 发现日期：2026-09-08。
  - 问题：当前 AIME24 30 题、AIME25 30 题、AMC23 83 题，共 143 个独立问题；`validation_n=16`
    只能降低同题解码方差，不能提供 16 倍的任务覆盖。在正确率约 30% 时，单个 30 题 AIME 集合的
    95% 区间仍约有 15--16 个百分点的半宽。
  - 改进：优先把统一 grader 的 held-out 数学评测扩展到至少 500 个、目标 1000 个独立问题，同时保留
    AIME24/AIME25/AMC23 分层结果。冻结题目清单、去重规则、prompt 模板、生成参数和随机种子；防止与
    DAPO-Math-17k 训练集污染。增加独立题目优先于仅增加同题 rollout 数。
  - 验收：逐题原始输出和评分可追溯；报告 unique prompt 数、每题 rollout 数、解析率、数据集分层
    accuracy/pass@k，以及 prompt-cluster bootstrap 95% CI；给出训练集去重或污染审计记录。
  - 进展：2026-09-10 已恢复并转换 MATH-500（500 题）、Minerva（272 题）和 Olympiad-Bench
    （675 题），与原有 143 题组成 1590-prompt 扩展评测；配置为每题 4 次、31,744-token 上限，共
    25,440 次四模型生成。prepare-only 已核验全部模型、prompt 长度和预计分片；规范化精确重合审计
    显示 DAPO-Math-17K 训练集重合 0/1590、评测内部重复 0，但该审计不能排除语义改写。正式生成与
    结果仍待运行，因此本条不勾选。

- [ ] **IMP-022（P1）建立 GPU 数量与 response 长度的短程算力标定矩阵。**
  - 发现日期：2026-09-08。
  - 问题：目前只有 2 GPU、7168 response token 的完整实测；4/8 GPU 加速比例以及 8192/12288 等更长
    response 预算的显存和吞吐没有证据。RTX 6000 Ada 通过 PCIe 通信，不能假设 GPU 数翻倍就线性加速。
  - 改进：使用固定数据、固定 seed 和同一训练条件，各运行 5--10 个代表性训练步；在资源允许时比较
    2/4/8 GPU，并比较 7168/8192/12288 response 上限。记录生成、teacher scoring、ROPD support、actor
    update、checkpoint 各阶段耗时，以及 token throughput、CPU 内存、GPU 峰值和 OOM 情况。长程实验
    只能在短程结果完成后排期。
  - 验收：生成一份可机器读取的 benchmark 表和可读报告；给出每个正式条件的预计 wall time、GPU-hours、
    存储预算和安全启动阈值，并明确实测值与外推值。
  - 证据：当前基准为控制臂 279 步、128488.7 秒、约 4.17 亿 token；待补充多 GPU/多长度结果。

- [ ] **IMP-023（P1）验证跨模型和跨数据分布的可迁移性。**
  - 发现日期：2026-09-08。
  - 问题：当前训练结论只来自 DeepSeek-R1-Distill-Qwen-1.5B 学生、JustRL-DeepSeek-1.5B 教师和单一
    数学训练域；即使多 seed 显著，也不能直接声称是普遍的 OPD 规律。
  - 改进：在 IMP-004、IMP-003/021 和 IMP-016 给出稳定主效应后，至少增加一组能力差异更明确的师生
    组合，并增加一个预先声明的分布外或不同难度数学评测层。7B 或更大模型只能先做短步显存 probe，
    不因名义总显存直接启动完整训练。
  - 验收：不同模型对和数据层使用一致的主要终点与配对对照；分别报告效应量和置信区间，并区分
    “同模型族复现”“跨规模迁移”“跨分布迁移”。
  - 证据：待补充。

- [ ] **IMP-024（P2）清理训练完成后的 Ray/SwanLab 退出异常。**
  - 发现日期：2026-09-10。
  - 问题：seed 42 的 OPD 和 ROPD 均成功完成并写出最终 checkpoint，但退出阶段日志出现 SwanLab
    `RuntimeError: cannot join current thread`；ROPD 日志还夹杂 Ray DataLoader worker 在结束阶段收到
    `Killed` 信号的 traceback。当前外层进程仍以退出码 0 完成，不影响已落盘训练结果，但会污染错误
    监控并可能掩盖未来真实故障。
  - 改进：在 Ray worker 关闭前显式、幂等地 finalize logger，避免从 tracking 析构函数所在 consumer
    thread 再调用 `finish()`；区分预期 worker shutdown 与训练中 worker failure。
  - 验收：至少一个短 smoke run 在保存最终 checkpoint 后无上述 traceback；重复 finalize 不报错；
    真正的数据 worker 异常仍能传递非零退出码并写入 `status.yaml`。
  - 证据：上述两个完整 run 的 `logs/train.log` 尾部；修复 commit 待补充。
    2026-09-14 完成的 FP32/offload 原始 OPD run
    `20260913_053418_opd_fp32_offload_seed42_216ed12` 在最终 checkpoint 已完整写入且外层退出码为 0 后，
    仍复现 `cannot join current thread` 和预期关闭阶段 DataLoader worker 收到 `Killed` 的 traceback；
    再次确认这是独立的退出清理问题，训练产物本身不受影响。

- [ ] **IMP-025（P0）用更新幅度匹配的对照分离“可靠性门控”和“整体缩小学习信号”。**
  - 发现日期：2026-09-10。
  - 问题：seed 42 完整处理臂的平均 OPD token reward 为约 -0.256，门控后 ROPD reward 为约
    -0.048；actor `grad_norm` 也从控制臂 1.812 降到 0.323。当前 ROPD 因此近似同时施加了选择性门控
    和约 80% 的全局更新衰减。若只比较原始 OPD 与当前 ROPD，任何性能差异都可能来自等效学习率变化，
    不能归因于教师可靠性判断。
  - 改进：在不使用最终测试标签的 frozen calibration 数据上确定全局缩放系数，增加
    `scaled-OPD`（所有 token 统一缩放到与 ROPD 相近的 reward RMS/梯度范数）对照；同时评估一个保持
    ROPD token 相对权重但把整体 RMS 恢复到 OPD 水平的 `normalized-ROPD` 消融。先短程确认 update norm
    匹配，再决定是否完整训练。
  - 验收：至少比较 OPD、scaled-OPD、ROPD 和 normalized-ROPD；报告 reward RMS、gradient norm、
    checkpoint 的真实参数更新范数/相对变化、effective token mass、训练稳定性和统一 held-out
    correctness。由于 Adam 的二阶矩归一化会部分抵消统一 reward 缩放，不能只凭 reward RMS 或原始
    gradient norm 宣称“更新幅度匹配”；若 scaled-OPD 的实际参数变化仍接近原始 OPD，应改用学习率匹配
    或其他能匹配 optimizer step 的对照。只有 ROPD 在实际更新幅度匹配后仍优于全局缩放，才把收益归因
    于选择性可靠性门控。
  - 进展：2026-09-10 已实现 `opd`、`scaled_opd`、`ropd`、`normalized_ropd` 四种训练 reward mode；
    normalized-ROPD 按 batch 把 scalar token reward RMS 恢复到 OPD 水平，并记录缩放、RMS、clip 与
    degenerate 指标。新增 seed 43 四臂配置、自动选择 2--4 张空闲 GPU 的顺序 tmux 启动器和 CPU
    测试；所有配置 dry-run 通过。2026-09-11 发现原先 `fixed_opd_scale=0.18` 来自 BF16 近乎无效训练，
    不能直接冻结为正式 FP32 对照的校准值；必须先用不读取最终测试标签的 FP32 短程 calibration 重新估计，
    再完成 5 步四臂 probe 并验证梯度/RMS。此前不要启动完整四臂训练，故不勾选。
    2026-09-15 已完成下一阶段代码准备：旧 `0.18` 被从配置中移除；启动器会先以独立 seed 1043
    运行 FP32 OPD calibration，只从每步 `ROPD token RMS / OPD token RMS` 的中位数冻结全局尺度，并保存
    输入指标哈希和校准 JSON，随后才以 seed 43 顺序运行 OPD、scaled-OPD、ROPD、normalized-ROPD
    四臂 5-step probe。直接启动未校准的 scaled-OPD 会被配置校验拒绝。下一验收点是 probe 的 reward
    RMS、actor gradient norm 和 effective token mass，而不是 correctness。
    该探针已于 2026-09-15 07:11 UTC 全部完成，编排退出码 0；calibration 和四臂均为 5/5 step，
    每个 run 均有完整的两个 model/optimizer rank shard。冻结尺度为 0.424445。OPD/scaled-OPD/ROPD/
    normalized-ROPD 的平均 selected reward RMS 分别为 0.6172/0.2628/0.2613/0.6248，说明两个 RMS
    对照达到预期；但平均 actor grad norm 分别为 1.9590/0.8295/0.3166/0.7545，说明 reward RMS
    匹配不等于梯度匹配。下一步必须审计四个 step-5 checkpoint 的真实参数变化，并先解决 IMP-007 的
    `lambda=1` 门控塌缩，不能直接启动完整训练。
    已准备可恢复的四臂 checkpoint 审计器，按代表性 attention/MLP/norm 参数报告真实 delta RMS、L2
    相对变化及相对 OPD 比率。该审计已于 2026-09-15 完成：OPD/scaled-OPD/ROPD/normalized-ROPD 的
    代表性参数 `delta RMS` 相对 OPD 分别为 `1.000/0.998/0.959/0.970`，尽管对应平均 gradient norm
    相对值仅为 `1.000/0.423/0.162/0.385`。这表明 Adam 在短程内几乎抵消了统一 reward 缩放，reward
    RMS 或 gradient norm 匹配都不足以构成“实际更新匹配”；下一对照应直接匹配 optimizer 后的参数
    delta（优先通过学习率校准），故本条仍不勾选。
  - 新增控制（2026-09-15）：除学习率校准的 OPD 对照外，加入 `shuffled-gate`。它在同一 batch/prompt
    分层内打乱 trust 与状态的对应关系，保持 trust 直方图、零门控率和有效 reward mass 尽量一致，但
    破坏“可靠状态得到更高权重”的语义。如果真实 gate 优于 shuffled-gate，且两者 checkpoint 参数
    delta 可比，才构成选择性可靠性信号优于单纯稀疏/缩放效应的证据。
  - 证据：seed 42 配对 run 的 `metrics/ropd_step_metrics.jsonl` 与 `logs/train.log`；实现和 probe 配置
    位于 `verl/verl/trainer/ppo/robust_opd.py`、`configs/experiments/opd_update_matched_seed43_*.yaml`。

- [x] **IMP-027（P1）清理正式评测最终状态中的过期阶段字段。**
  - 发现日期：2026-09-10。
  - 问题：本次评测完成后的 `status.yaml` 正确标记了 `status: completed`、`stage: complete`，但仍保留
    上一阶段的 `model: ropd_step279`、`completed_shards: 0`、`total_shards: 48`。实际结果是该模型
    48/48 shards、2288/2288 条均已完成；过期字段会使自动状态检查误报。
  - 改进：阶段切换和最终完成时显式清除瞬态的 model/shard 字段，或把阶段进度移入独立的嵌套结构；
    状态查看器应以产物清单交叉校验聚合状态。
  - 验收：完成态只显示最终聚合计数，失败/恢复态仍能准确显示当前模型和 shard；增加状态转换 CPU 测试。
  - 完成证据：2026-09-10 的 `update_status(clear_fields=...)` 已在 merge、analysis、complete 转换中
    清除 model/shard 瞬态字段，完成态同时清除旧 error/traceback；CPU 状态转换测试验证最终字段不存在。
    既有冻结 run 的历史 `status.yaml` 不回写，新运行自动采用修复。

- [x] **IMP-028（P1）为正式评测补充配对的 pass@k 差值置信区间。**
  - 发现日期：2026-09-10。
  - 问题：当前看板展示各模型 pass@k 点估计/边际区间，但主要比较表只对 Avg@16（等价于总体
    pass@1）计算配对差值 CI。ROPD 相对 OPD 的 pass@16 点估计高 3.50 pp，若没有配对差值区间容易
    被误读为可靠提升；本轮补算的 95% CI 为 `[-0.70,+8.39] pp`，仍跨过 0。
  - 改进：在正式分析脚本中按 prompt 进行配对 bootstrap，输出每个预先声明的 k 的模型差值、95% CI
    和方向翻转计数；明确主要终点与探索性终点，避免事后选择最有利的 k。
  - 验收：CSV、JSON、SVG/HTML 同时包含配对 pass@k 差值和区间，CPU 测试覆盖配对样本顺序与缺失值。
  - 完成证据：2026-09-10 正式分析器已对所有预声明 k 输出配对差、prompt-bootstrap 95% CI 和逐题
    胜/平/负计数，生成 `paired_pass_at_k_deltas.svg`、CSV、JSON 与 HTML；随机流按指标命名，防止新增
    统计改变旧 CI。合成 CPU 分析测试覆盖了 pass@k 配对列和 SVG 产物。

- [x] **IMP-029（P0）按论文 31,744-token 评测口径复核原始 OPD 的提升幅度。**
  - 发现日期：2026-09-10。
  - 问题：当前 `OPD−initial=+1.75 pp` 来自 7168-token 评测，初始/OPD 约 60.45%/58.52% 的输出
    达到上限；论文正式评测是每题 16 次、temperature 0.7、top-p 0.95、31,744-token response 上限，
    并报告同一师生组合回收超过 80% 的教师—学生 gap。两个数字不能直接比较。
  - 改进：使用初始学生、JustRL 教师、seed 42 OPD/ROPD step 279 四模型，在原论文 143 题上完成
    9152 次论文口径生成；按数据集报告 avg@16，并以配对 prompt bootstrap 计算训练后提升和 gap
    recovery。先复核现有 checkpoint，不能因为短预算结果偏低就直接重训。
  - 验收：四模型全部分片完整、配对 seed 无误；parse/at-limit/完整率分开报告；教师 gap 为正时给出
    recovery 点估计与 95% CI，并和论文“超过 80%”作同口径讨论。
  - 完成证据：`20260910_165940_opd_ropd_seed42_formal_evaluation` 于 2026-09-11 03:06 UTC 以退出码
    0 完成；四模型均为 48/48 shards、2288/2288 条，共 9152 条生成，分析和看板齐全。31,744-token
    口径下初始学生/教师/OPD/ROPD average correctness 为 48.95%/66.56%/48.82%/50.70%；教师—学生
    gap 为 17.61 pp，OPD gap recovery 为 -0.74%（95% CI `[-11.16%,8.65%]`），ROPD 为 9.93%
    （95% CI `[-0.23%,19.24%]`）。OPD 并未复现论文所报超过 80% gap recovery，满足 IMP-030 的
    条件性数值/训练配置复核触发条件。

- [x] **IMP-030（P1）仅在长预算评测仍异常时复核 fp32/8-GPU 论文训练数值设置。**
  - 发现日期：2026-09-10。
  - 问题：当前完成的控制臂采用 bf16、2×RTX 6000 Ada 和 CPU offload；论文默认训练配置是 fp32、
    8×A800。目标函数、batch、rollout、top-k、学习率和 epoch 已一致，但数值精度与硬件路径尚非完全
    同口径。
  - 改进：将该项设为条件分支。只有 IMP-029 仍显示 OPD gap recovery 显著偏低，才先运行 fp32、
    8-GPU、无 actor 参数/优化器 offload 的一步 full-shape probe；probe 成功后再决定是否完整训练。
  - 验收：probe 记录峰值显存和阶段耗时且无 OOM/NaN；如启动完整训练，必须再做 31,744-token 独立
    评测并与 bf16/offload 运行配对比较。不可仅凭训练 loss 判定复现。
  - 进展：`opd_paper_exact_8gpu_probe.yaml`、`opd_paper_exact_8gpu.yaml` 已通过 dry-run；IMP-029 的
    正式结果显示 OPD gap recovery 约 -0.74%，显著偏离论文“超过 80%”的报告，因此该条件分支已经
    触发。2026-09-11 对既有 seed-42 OPD checkpoint 的审计发现，actor 的 17.77 亿参数和 Adam
    `exp_avg/exp_avg_sq` 均为 BF16；279 步后抽查层仅 0.98%--1.21% 元素改变，`model.norm.weight`
    改变比例为 0%。对应训练日志的 top-k overlap 也只从首步 72.13% 到末步 73.56%，前后 20 步均值
    仅增加 1.31 pp。这构成“低学习率更新被 BF16 量化吞掉”的直接证据，而非单纯评测噪声。
    当前 FP32 offload 和 8-GPU paper-exact probe 已统一改为 FP32并通过 dry-run。两卡 FP32/offload
    一步 full-shape probe `20260911_112551_opd_fp32_offload_probe_seed42_693bda4` 已于 2026-09-11
    以退出码 0 完成，单步训练约 512.6 秒、含初始化和保存总计 652.9 秒；actor/optimizer 两个 rank
    checkpoint 完整，无 OOM/NaN。它证明低显存执行路径可行，但不是 8-GPU/no-offload 精确执行路径，
    因而本条仍不勾选。FP32 修复、审计器和正式评测代码已提交为 `d53852f`。两卡 FP32/offload 完整
    原始 OPD run `20260913_053418_opd_fp32_offload_seed42_216ed12` 已从后继提交 `216ed12` 完成
    279/279 步，退出码 0，总耗时 117881.4 秒（约 32.74 小时），无 OOM、NaN 或 NCCL 错误；最终
    `global_step_279` 的 model/optimizer 两个 rank shard、extra state、tokenizer/config 和 `data.pt`
    均完整。下一验收步骤是对该 FP32 checkpoint 运行固定 31,744-token 独立评测并与初始学生及旧
    BF16 checkpoint 配对比较；在评测完成前不可仅凭训练日志宣称复现论文提升，因此当时未勾选。
    2026-09-14 已新增 `opd_fp32_seed42_paper_aligned_evaluation.yaml`，主要比较预注册为
    `opd_fp32_step279-initial_student`，并保留 FP32-vs-BF16 与教师 gap recovery；真实 preflight 确认
    143 题、每题 16 次、总计 9152 条，其中 6864 条严格复用、仅 2288 条需要新增生成。
  - 完成证据：`20260914_155417_opd_fp32_seed42_paper_aligned_evaluation` 于 2026-09-14 17:48 UTC
    以退出码 0 完成。四模型均为 48/48 shards、各 2288 条，配对 seed、prompt manifest 和生成配置
    校验通过，无 OOM/NaN/NCCL 错误。初始学生、教师、旧 BF16 OPD、新 FP32 OPD 的平均正确率分别为
    48.95%、66.56%、48.82%、63.11%；FP32 OPD 相对初始学生提升 14.16 pp，prompt-bootstrap 95% CI
    `[11.10,17.40]` pp；教师 gap recovery 为 80.40%，95% CI `[71.15%,90.22%]`。FP32 相对 BF16
    提升 14.29 pp，95% CI `[11.06,17.61]` pp。两卡 offload 路径已复现论文“超过 80%”的量级，因此
    当前没有必要为了结论再强制执行 8×A800/no-offload；硬件精确复刻仅保留为需要时的工程对照。

- [ ] **IMP-031（P1）让正式评测的运行中 shard 进度实时、原子地回写状态。**
  - 发现日期：2026-09-11。
  - 问题：论文口径评测执行到 `ropd_step279` 实际已有 38/48 shards、1458/2288 条生成落盘时，
    `status.yaml` 仍停留在该模型开始时的 `completed_shards: 0`。`--status` 会交叉扫描产物并正确显示
    38/48，但只读取 YAML 的监控器会误判进度。
  - 改进：父进程等待 generation workers 时定期扫描原子 shard，并把当前模型、完成 shard/生成数和
    最近进展时间原子写入状态；不要让多个 worker 并发写同一个 YAML。
  - 验收：短 synthetic worker 测试中进度单调递增，完成态仍满足 IMP-027 的字段清理规则；worker
    失败或 shard 损坏时，YAML 与 `--status` 扫描结果一致。
  - 证据：`20260910_165940_opd_ropd_seed42_formal_evaluation` 在 2026-09-11 02:29 UTC 的状态核对。
  - 进展：2026-09-14 父进程已改为生成期间每 30 秒扫描原子 shard 并回写
    `completed_shards/total_shards`、heartbeat 和最近进展时间；结束阶段会清理这些瞬态字段。CPU
    测试与真实复用 preflight 均通过，待本次 FP32 正式评测提供实际长任务证据后决定是否勾选。

- [x] **IMP-032（P0）阻止低精度 actor 造成静默无效的 OPD 训练。**
  - 发现日期：2026-09-11。
  - 问题：RTX 适配配置曾把 `models.dtype` 改为 BF16，但当前 verl/FSDP 路径没有保留 FP32 master
    parameters 或 FP32 Adam moments。进程可以正常训练和保存 checkpoint，却几乎不产生有效的
    `1e-6` 参数更新，因此仅看退出码、梯度范数或 checkpoint 存在会误判复现成功。
  - 改进：所有科学训练配置恢复 FP32；managed launcher 对非 FP32 actor 默认报错。只有纯 plumbing
    smoke config 可通过 `allow_low_precision_actor_training=true` 显式豁免，同时输出不可用于效果结论
    的警告。增加 checkpoint dtype/参数变化审计器与 top-k 奖励符号单测。
  - 验收：论文精确和 4-GPU offload 配置 dry-run 均生成 FP32 actor；BF16 普通配置被拒绝、显式 smoke
    豁免被警告；CPU 测试验证 top-k OPD 梯度使 reverse KL 下降。
  - 完成证据：`scripts/run_opd_experiment.py`、`scripts/audit_opd_reproduction.py`、
    `verl/tests/trainer/ppo/test_opd_reproduction_audit_on_cpu.py`；8 个 CPU 测试通过，paper-exact dry-run
    显示 `model_dtype=fp32`、`ppo_max_token_len_per_gpu=32768`、reward microbatch 24，4-GPU offload
    dry-run 显示 FP32 actor 与参数/优化器 offload 均启用。上述两卡 probe 的 checkpoint 审计进一步
    确认 actor 17.77 亿参数和 Adam moments 全部为 FP32；一步后抽查 attention/MLP/norm 参数的变化率
    为 99.95%--100%，平均绝对变化约 `0.74e-6`--`0.99e-6`，与旧 BF16 运行的 0%--1.21% 形成直接对照。
    实现提交：`d53852f`。

- [ ] **IMP-033（P1）冻结顺序多臂实验的代码版本。**
  - 发现日期：2026-09-15。
  - 问题：本次四臂探针运行期间仅提交了 `docs/IMPROVEMENT_CHECKLIST.md`，因此 calibration、OPD、
    scaled-OPD 的 manifest 指向 `3c2300c`，ROPD、normalized-ROPD 指向 `2a1bb46`。两个 commit 的唯一
    差异是文档，训练代码和配置没有变化，所以不影响本次方法比较；但 manifest commit 不统一会降低
    自动审计清晰度，未来若运行代码发生变化则可能破坏严格配对。
  - 改进：顺序多臂启动器在开始时冻结 HEAD 和工作树状态，并在每个实验臂启动前复核；出现任何 tracked
    代码/config 变化时停止，或改为从固定 commit 的独立 worktree 启动。运行中的纯结果/日志写入不应触发。
  - 验收：合成测试验证未变化时所有 arm 使用同一 revision，文档或代码发生提交/修改时在下一 arm 前
    明确停止；run manifest 记录冻结 revision 和校验结果。
  - 证据：本次 `runs.tsv` 与 `git diff 3c2300c..2a1bb46`；实现待补充。
  - 进展：update-matched 四臂启动器和新 LCB 诊断启动器均已在起始时要求 clean worktree、冻结 HEAD，
    并在每个后续阶段前复核；Shell 语法测试通过。仍需补充模拟中途变更的自动化测试后再勾选。

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
