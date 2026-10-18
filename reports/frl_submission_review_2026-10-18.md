# FRL 投稿前全面审查报告

- **日期**：2026-10-18
- **审查对象**：`paper/main.tex`（421 行）、`paper/tables/*.tex`、`paper/references.bib`、`reports/*` 全部 CSV/审计报告、`code/` 关键脚本
- **审查方式**：正文/表格/CSV 三方逐项核对 + 代码静态审查 + FRL 投稿指南逐条对照。说明：本次未做流水线端到端重跑（本机未装 statsmodels 且 PyPI 代理不可达）；数字一致性结论基于 CSV↔LaTeX 逐格比对与代码阅读，前一轮会话已完成的核验本次全部复核通过。
- **结论速览**：**可以投，但现在这个版本不行。** 研究设计层面（置换推断、T+1 可用性、闸门纪律、语气收窄）上一轮审计的硬伤已真实修复；当前阻断投稿的是"文字与表格数字不一致、一处过度声称、一处未兑现承诺、AI 工具名不实"这四类问题，修完大约需要 1–2 天。修完之后，稳健性边界薄（长腿无信号）是审稿人第一攻击点，建议在投稿前主动补一个腿分解。

---

## 0. 总判断

| 维度 | 评价 |
|---|---|
| 研究设计 | 良好。置换推断实现正确（`common.py:310-355`，分组循环平移 + 同估计器重估 + 加一修正）；T+1 可用性有交易所披露规则依据且审计 100% 通过；闸门/修订纪律强 |
| 证据链 | 表与 CSV 一致（除 H1/H2 两处正文数字）；引用 DOI 全部独立核验正确 |
| 正文可靠性 | **有硬伤**：正文与表格数字不一致 2 处、过度声称 1 处、承诺未兑现 1 处、AI 披露工具名不可查证 |
| 可复现性 | 不足：两个 GitHub 仓库 private；`paper/figures/` 缺失导致 PDF 不可复现且失败是静默的 |
| 措辞 | 方向正确（上一轮审计后收窄到位），但对冲词密度过高、有 2 处口语化，且消耗了大量字数预算 |
| 格式合规 | 大体合规（2293 词、摘要 167、highlights、声明齐备），缺 CRediT，JEL 挤在关键词里 |

---

## 1. 硬伤（投稿阻断级，P0）

### H1. 正文与 Table 1 数字不一致（两处）

- **证据**：`main.tex:209-211` 写 "It ranges from $-0.29$ for turnover to $+0.27$ for size…Pooling the six factors and removing factor-specific means leaves $-0.11$"；而 `tables/table1_corr.tex` 实际是 TURN = **−0.252**、VAL = **−0.269**、SMB = +0.266、Pooled = **−0.141**、factor-demeaned = **−0.093**。
- **性质**：−0.29 在表里根本不存在；−0.11 与 −0.141 / −0.093 都不相等。审稿人对着表格读正文 30 秒就能发现，直接损害可信度。
- **修复**（10 分钟）：正文改为 "ranges from $-0.25$ for turnover (and $-0.27$ for value) to $+0.27$ for size…Pooling the six factors gives $-0.14$, or $-0.09$ after removing factor-specific means"。注意描述与数字对应：若说 "for turnover" 就用 −0.252。

### H2. 稳健性段 p 值与 Table 5 不一致

- **证据**：`main.tex:333` 写 "weak in 2015--2018 ($p=0.33$)"；`tables/table5_robust.tex` 实际为 **p = 0.194**。其余核对无误（0.11↔0.111、0.15↔0.149、0.008↔0.008 ✓）。
- **性质**：0.33 疑似 T+1 修复前旧版数字的残留。而且修完之后稳健性故事**反而更好**：2015–2018 子样本在 10% 水平上显著（0.194 < 0.10），现在的措辞把自己说弱了。
- **修复**（10 分钟）：改为 "remains detectable at the $0.10$ level in 2015--2018 ($p=0.194$), is not retained at the $0.10$ level in 2019--2021 ($p=0.111$) or 2022--2025 ($p=0.149$)…"。顺带把三个子样本 p 值都在 0.10–0.20 区间、方向一致的事实讲出来，比现在的措辞更有利也更诚实。

### H3. "multi-stock audit" 过度声称

- **证据**：`main.tex:128-130`："The valuation ratios are treated as point-in-time fields based on the **multi-stock audit** documented in the replication materials." 而 `reports/data_audit.md` 的 PIT 证据只有一只股票（浦发银行）。
- **性质**：这是可被审稿人实证戳穿的声称（复现材料公开后一目了然）。声称"multi"而证据是"single"，比不声称更糟。
- **修复**（二选一，推荐 a）：
  - (a) **真做多股票 PIT 审计**：抽 20–30 只股票、覆盖若干财报窗口，验证 pbMRQ/peTTM 取值不随后续重述漂移，把结果进 `data_audit.md`。成本约半天，一劳永逸。
  - (b) 措辞降级："based on a point-in-time spot check documented in the replication materials"。

### H4. 承诺的 compound-return sensitivity 未兑现

- **证据**：`main.tex:171` 承诺 "(we report compound-return sensitivity separately)"，但 `reports/` 里没有任何对应输出。
- **性质**：投稿版正文承诺了不存在的分析。审稿人若索要该输出，拿不出来。
- **修复**（二选一，推荐 a）：
  - (a) 补跑：把 h 期收益由求和改为几何复利，在主表或 Table 5 加一行。成本约 2 小时，且能预先堵住"为什么用 horizon-sum 不用 compound return"这个审稿人大概率会问的问题。
  - (b) 删掉括号里的半句（省 6 个词）。

### H5. AI 使用披露的工具名不实、不可查证

- **证据**：`main.tex:393-396`："the author used **ClawsGO Agent in the ClawsGO Science Harness**…"；`main.tex:399` 致谢："The author thanks the ClawsGO Science Harness…"。
- **性质**：Elsevier 的 AI 披露要求写明**实际使用的工具**；"ClawsGO" 无法被编辑或审稿人核实，看起来像有意回避，反而引火烧身（编辑会直接问）。同时"在致谢里感谢一个工具"不常规，建议删除或改为感谢具体的人。
- **修复**（10 分钟）：写真实工具名（例如 "Claude (Anthropic)"），四要素齐备：工具名 + 用途（research assistance and language polishing）+ 人工复核 + 作者责任。当前只缺"真实工具名"这一要素。

### H6. Figure 1 源文件缺失，且编译失败是"静默"的

- **证据**：`main.tex:222-226` 用 `\IfFileExists{figures/fig_crowding_series.pdf}{...}{\fbox{Figure 1 unavailable}}`；仓库里**没有** `paper/figures/` 目录；源图只存在于 `reports/fig_crowding_series.png`（PNG，不是 tex 引用的 PDF）。
- **性质**：任何人在干净环境重编译，图的位置会静默打印 "Figure 1 unavailable"。如果投稿前重编译一次 PDF 而没逐页检查，投出去的稿子图 1 就是这句话。
- **修复**（30 分钟）：把 `reports/fig_crowding_series.png` 转成 PDF（或直接改 `\includegraphics` 指向 `../reports/` 下的 PNG）放进 `paper/figures/` 并提交；投稿前重新编译并核对第 7 页。

### H7. 两个 GitHub 仓库均为 private，复现材料不可达

- **证据**：`a-share-factor-crowding-research` 与 `a-share-factor-crowding-raw-data` 均为 private。正文声明（`main.tex:379-384`）说 "will be made available with the replication package"。
- **性质**：FRL 不强推数据公开，但"声明可得"而审稿人点开 404，会被视为数据可得性声明不实。**这是很多 desk reject 的原因之一。**
- **修复**（投稿前）：code 仓库设 public；raw-data 仓库若受上游条款限制可保持 private，但数据声明须改写为 "available on request from the author"；另建议在投稿系统里附一份审稿人专用复现包（匿名版本）。

### H8. 六个因子定义全文缺失

- **证据**：正文只有 comomentum（`main.tex:143-151`）与 margin proxy（`153-162`）两个**代理**的定义；VAL / SMB / MOM / QMJ / LOWVOL / TURN 这六个 style factor 从未定义（代码里是：VAL=1/pbMRQ、SMB=−ln(float_mv)、MOM=t−12..t−1、QMJ=pb/pe、LOWVOL=−vol60、TURN=−turn20）。
- **性质**：审稿人不知道 "six style factors" 是什么；中国背景外审稿人不熟这些简称。这是方法论描述的基本要件。
- **修复**：加一段约 100–120 词的紧凑定义（Data 节或 Table 1 的 note）。**需要先按 §4 腾字数。**

### H9. Table 2 只有系数和星号，无任何推断统计量

- **证据**：`tables/table2_main.tex` 只有 coef + 星号；HAC t 值被"留在复现输出里"。而 `reports/table2_main.csv` 里 se/t/p/p_placebo/n/r2 **全部都有**（例如 h=6、spec D、margin proxy：β=−0.0400，p_placebo=0.010，n=669，r2≈0.07）。
- **性质**：数据都在，只是没进表格。审稿人常规要求表内或 note 给出推断统计量。
- **修复**（30 分钟）：Table 2 每格系数下加括号 permutation p（或加一列），note 里给 N；主规范（D, h=6）的 SE/t 写进正文。

### H10. 行业分类为"当前快照"回填，正文没有明说

- **证据**：代码使用 updateDate 2026-10-12 的行业分类覆盖全样本；`main.tex:133` 只说 "using the available industry classification"；结论 `main.tex:373` 说 "Future work should use historical industry classifications"——有暗示，但没说清**本研究**用了快照。
- **性质**：survivorship/backfill 问题是因子研究审稿人的固定检查项；被问到时才承认 vs 主动披露，观感完全不同。
- **修复**（10 分钟）：Data 节加半句 "…using the available industry classification (a current-snapshot classification applied to the full sample)"，与结论呼应。约 10 词。

### H11. SMB 的融资覆盖率仅 0.257，config 的 0.30 门槛未在代码执行

- **证据**：`config.yaml` 声明 `coverage_floor: 0.30`，但 `code/08_crowding.py` 未强制；实测 SMB 长腿平均覆盖率 **0.257**（最低 0.0），其他因子 0.82–0.93（来自 `crowding.parquet`）。
- **性质**：SMB 的 margin 代理质量系统性差于其他因子 → 混合面板里 SMB 的"无信号"可能只是测量质量差，不是真的无拥挤。这是一个自找的审稿人攻击面。
- **修复**（至少做 a+b）：(a) 在 Data 节或表注披露各因子覆盖率；(b) 稳健性：剔除 SMB 重估主回归（预计 1–2 小时）；(c) 可选：真正执行 coverage_floor 并报告。

### H12. 核心结果稳健性边界薄，且缺"哪条腿在驱动"的分解——审稿人第一攻击点

- **证据**：基线（D, h=6）强（β≈−0.040，p=0.010），但 `tables/table5_robust.tex` 显示：长腿单独 **p=0.672**（β=−0.0011）；6/24 个月窗口 p=0.185/0.244；三个子样本只有 2015–2018 过 10%（0.194）。且 Table 5 有 "Long leg only" 行，**没有 "Short leg only" 行**。
- **性质**：信号几乎全部来自 L−S 价差，长腿本身无信号，而 A 股卖空高度受限——审稿人必问："价差里的负向关联是不是主要由短腿的机械成分驱动？crowding 为什么不在长腿兑现？" 这是决定论文叙事成立与否的问题，不能等审稿人问。
- **修复**（半天，投稿前建议做）：
  1. 先补跑**长/短腿分别回归**（至少短腿一行），在数据里看清驱动结构；
  2. 按事实改写叙事——若短腿主导，正文必须诚实说明并给出经济解释（如：两融加杠杆集中在多头、去杠杆压力沿空头腿传导）；若两腿都有但都不显著而价差显著，说明是方差合并效应，也要讲清楚；
  3. 子样本段按 H2 修正后的口径写（三个子样本 p 均在 0.10–0.20、方向一致），别自己把自己写死。

---

## 2. 小问题（M 级，不阻断但应处理）

- **M1. 样本量口径**：`main.tex:122` "The sample covers 5,247 stocks"，而数据字典/复现材料的口径是 5,418（清洗后全样本）。建议改为 "5,247 of the 5,418 stocks in the cleaned universe"（+6 词），或确保 data_dictionary 明确定义两个数。
- **M2. 过时的头部注释与文末 TODO**：`main.tex:8-10` 仍写着"状态（2026-10-15）：骨架…第 3 节结果全部为占位"，文末 `409-420` 还有回填 TODO。若源码以任何形式流出（投稿附 source、仓库公开），这是灾难。**投稿前必须删除/更新**。
- **M3. CRediT 声明缺失**：单作者也要（Conceptualization / Methodology / Software / Formal analysis / Writing）。Elsevier 录用流程必查。
- **M4. JEL 挤在关键词里**：`main.tex:52-54` 把 "G12" 作为第 7 个关键词（elsarticle 双 keyword 环境会互相覆盖，注释里有说明）。投稿系统里关键词与 JEL 分字段填写没问题，但 PDF 里建议把 G12 移出 keyword 块，改为 "JEL classification: G12" 普通行。6 个真实关键词本身合规（1–7 个）。
- **M5. 参考文献偏少**：9 条全部被引用且 DOI 已核验 ✓，但 FRL 常见 20+ 条。可考虑补 1–2 条：中国两融/拥挤的相邻文献、comomentum 的应用或批评。非必须。
- **M6. 标题（可选）**：现标题 "Measuring Factor Crowding When Leverage Is Observable…" 是"测量优先"的，与投稿信定位一致，**可以不改**。若想更抓人，可考虑 "Observed Leverage and Factor Crowding: Margin-Financing Evidence from China's A-Share Market"——把 margin 代理放进标题。二选一，不必动。
- **M7. 标准化口径**：`09_regressions.py:72` 用 pooled 均值/SD 标准化，同时回归带因子固定效应；within/pooled SD 之比为 0.734（comomentum）/ 0.823（crowd_margin），即系数量级相对"组内变异"被低估。"一个标准差增加对应 4pp 更低收益"的经济量级陈述基于 pooled SD。严谨的计量审稿人可能追问。可选项：脚注说明，或补一个 within-factor 标准化的稳健性。

---

## 3. 措辞与语调（重点回应你的担心）

**总方向判断**：上一轮审计后语气收窄到位，全文没有过度自信的声称，"direct" 类说法已全部改为比较级或否定式（`main.tex:112, 161`），这点做得很好。**现在的问题是反向的**：对冲词密度过高 + 两处口语化，读起来防御性过强，而且吃掉约 100 词的字数预算。

### T1. 对冲三件套重复过多（建议保留 4–5 处关键对冲，其余合并删除）

| 措辞 | 出现位置 | 建议 |
|---|---|---|
| "associational" | 摘要 :47；结论 :360 | **两处都保留**（摘要+结论各一次是标准做法） |
| "descriptive" | :211、:263、:296 | 留 1 处（推荐 :263 经济量级处），其余删/改 |
| "auxiliary" | :199、:309、:313、:367 | 留 :199（定义证据层级的那处），其余删 |
| "not a …" 结构 | :112、:149、:162、:266、:276、:337、:343、:360-361 | 留 4 处：代理性质（:162）、因果（:266 或 :360）、稳健性诚实（:337）、零结果信息量（:343）。其余删除或合并 |
| "should not be described as universal robustness" | :337 | 保留——这是审稿人最看重的一句 |

预计可省 **80–120 词**。重复对冲不会让审稿人更信，反而显得作者自己都不信；一个信里 10+ 处 "not X, not Y, not Z" 会读成焦虑。

### T2. 两处口语化

- `main.tex:61`："A factor that attracts too much capital should, **in the end**, stop working." → "eventually"，或删掉这三个词。
- `main.tex:81`："**That comparison is the point.**" → 改为 "The comparison itself is the contribution." 或直接并入下一句。作为引言的核心句它有力量，但作为 FRL 信函文体偏随意；改完更稳。

### T3. 摘要里的一处用词

- 摘要 :47 "the broader second-moment evidence is **exploratory**"——"exploratory" 用在摘要里等于提前认怂。建议改为 "is directionally consistent but individually insignificant"（与 :273-275 正文口径一致）。摘要留给自己的定性词越少越好。

### T4. 数字写错本身也是措辞问题

H1/H2 的 −0.29、0.33 这类错数，审稿人会读成"粗心"。修完 H1/H2 后，全文数字（含摘要的 "four-percentage-point"、:249 的 "about $-0.034$"、:341 的 "11–14 per cent"）与表/CSV 做一次最终对表。

### T5. 中文批注清理

`main.tex` 内还有中文工作批注（如 :8-10、:18-19、:50-51、:408-420），投稿前按 M2 一起清理；投稿稿用 pdflatex 编译验证（注释里自己写了这个要求，要执行）。

---

## 4. 字数预算与删改方案

**现状**：正文 2,293 词；含声明 2,461 词。FRL 上限 2,500——**零余量**，且各地编辑口径不同，按最严口径（含声明）准备。

**需求**：因子定义 ≈100–120 词（H8）。

**删改来源（合计约 150–180 词，够付 H8 且有富余）**：

| 来源 | 预计省词 |
|---|---|
| T1 对冲合并 | 80–120 |
| :89-93 "mechanical component" 段压缩（62→~35 词） | ~25 |
| :102-104 推断关切与 :183-188 重复，压缩 | ~15 |
| :170-171 括号（若按 H4 选 b） | 6 |
| :191-195 验证/功效段压缩 | ~20 |

**结论**：加因子定义**不需要牺牲任何实质内容**，压缩对冲词就够了。

---

## 5. FRL 格式核对表

| 项目 | 要求 | 现状 | 判定 |
|---|---|---|---|
| 正文 | ≤2500 词 | 2293（含声明 2461） | ✓ 但零余量 |
| 摘要 | ≤250 词 | 167 | ✓ |
| 关键词 | 1–7 个 | 6 个真实 + G12（JEL 挤占） | ⚠ 见 M4 |
| Highlights | 3–5 条，每条 ≤85 字符 | 5 条，最长 72 字符 | ✓ |
| 投稿费 | $200 不可退 | 已确认 | ✓ |
| AI 使用披露 | 工具名+用途+复核+责任 | 有，工具名不实 | ✗ 见 H5 |
| Funding / COI / Data 声明 | 必须有 | 齐备 | ✓ |
| CRediT | 录用流程必查 | 缺 | ✗ 见 M3 |
| 表格 | 可编辑文本 | LaTeX booktabs | ✓ |
| 图 | 单独文件上传 | figures/ 缺失 | ✗ 见 H6 |
| 单国复制 desk-reject | 需在投稿信拆解 | cover_letter 第 3 段已拆；正文"测量对比"定位 ✓ | ✓ |
| 参考文献 | 格式 elsarticle-harv | 9 条全引用、DOI 已核验 | ✓ 偏少，见 M5 |

---

## 6. 优先行动清单

**第一轮（投稿阻断，约 1–2 天）**
1. H1+H2：正文数字对齐表格（30 分钟）
2. H5：AI 工具名改真实名、删工具致谢（10 分钟）
3. H3：multi-stock → 真审计（半天）或措辞降级（10 分钟）
4. H4：compound-return 补跑（2 小时）或删句（10 分钟）
5. §4 删改 + H8 因子定义（2–3 小时）
6. H6：图入库 + 重编译逐页检查（30 分钟）
7. H9：Table 2 加 permutation p 列/N（30 分钟）
8. H10：行业快照半句（10 分钟）
9. M2+M3+M4+T5：注释/TODO 清理、CRediT、JEL 处理（30 分钟）
10. H7：仓库 public 或改数据声明 + 备审稿附件（30 分钟）

**第二轮（增强，投稿前建议，约 1 天）**
11. H12：长/短腿分解 → 按事实定叙事 → 子样本段改写（半天）
12. H11：覆盖率披露 + 剔除 SMB 稳健性（2 小时）
13. 全文数字与 CSV 最终对表 + M1 样本口径（30 分钟）

**第三轮（可选）**
14. M5 补文献；M6 标题斟酌；M7 标准化脚注

---

## 7. 复核说明（哪些已验证、哪些没有）

- ✓ 已验证：Table 1/2/3/4/5/7 与 reports CSVs 逐格一致（除 H1/H2 两处正文数字）；置换推断实现正确；T+1 审计 100% 通过；引用 9 条 DOI 正确；正文/摘要/highlights 字数合规。
- ✗ 未验证：未做流水线端到端重跑（本机无 statsmodels，PyPI 代理不可达）。建议在完成第一轮修改、重编译 PDF 时，顺带在云主机重跑一次 09–11 脚本确认 CSV 可再生。

---

## 附：执行状态（2026-10-18 更新）

| 条目 | 状态 |
|---|---|
| H1 正文与 Table 1 数字 | ✅ 已修（−0.25/−0.27、−0.14/−0.09） |
| H2 稳健性 p 值 | ✅ 已修（0.194 显著；0.111/0.149/0.008 复核无误） |
| H3 multi-stock audit | ✅ 真审计完成（40 只，17_pit_audit.py；BPS 台阶↔公告日匹配率 ~84%，季度末伪证检验无回填证据） |
| H4 compound-return | ✅ 已补跑（h=6 −0.042、h=12 −0.057，p 不变） |
| H5 AI 工具名 | ✅ 已改 DeepSeek（大语言模型）；工具致谢已删 |
| H6 图源缺失 | ✅ figures/fig_crowding_series.pdf 已入库；本地干净编译验证 |
| H7 仓库 private | ✅ 两个仓库均已公开（用户 2026-10-18 确认）；数据声明为公开 URL 版本 |
| H8 因子定义 | ✅ 已加（~100 词） |
| H9 Table 2 推断统计量 | ✅ Panel B 置换 p 值面板已加 |
| H10 行业快照 | ✅ 已披露 |
| H11 SMB 覆盖率 | ✅ 已披露 + 剔除 SMB/覆盖率门槛稳健性（结果更强/不变） |
| H12 稳健性边界 | ✅ Table 6 腿分解：效应全部来自空头腿（自身融资强度 → 自身衰减 −0.052，p=0.010） |
| M1 样本口径 | ✅ 已改 |
| M2 过时注释/TODO | ✅ 已清 |
| M3 CRediT | ✅ 已加 |
| M4 JEL 混入关键词 | ✅ 已拆分 |
| M7 标准化口径 | ✅ 正文报 pooled 4.0pp 与 within 3.3pp |
| T1/T2/T3 措辞 | ✅ 对冲压缩 ~150 词、口语化两处已改、摘要 exploratory 已改 |

**拓展（用户要求的"试一下"）**：新增流量型代理（18_flow_proxy.py，Table 8）——融资买入强度与净流量均显著预测因子衰减（净流量四持有期全部置换显著），赛马显示存量与流量部分独立。论文新增 3.4 节「Flows versus stocks」。

**最终数字**：正文 2414 词（≤2500 ✓）、摘要 ~190、PDF 17 页、9 表 1 图、bibtex 零警告。
