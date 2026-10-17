# 因子拥挤与 A 股因子衰减项目：全面审稿与复核报告

**审阅日期：** 2026-10-16（北京时间）  
**审阅主体：** ClawsGO Agent / ClawsGO Science Harness  
**审阅类型：** `structured challenge`——基于同一项目上下文进行的结构化红队审查，不应表述为独立盲审。  
**审阅范围：** `paper/main.tex`、参考文献、生成表图、`code/01–15` 关键脚本、配置、原始/清洗面板、两融数据、CSV 报告、数据审计、流水线脚本与项目规划文档。

---

## 0. 一页结论

### 总判断：研究问题有价值，数据工程已有相当基础，但当前稿件还不能送审

项目已经具备一篇有潜在发表价值的短文的核心材料：

- 问题不是简单复制“拥挤预测衰减”，而是比较**收益相关性间接测度**与**两融余额直接可观测代理**；
- 样本覆盖 2015–2025，初始股票池含退市股，当前原始日线与两融文件已实际落地；
- `ls_fwd1` 的时点错误已修复，因子组合收益与预测月份的主要对齐逻辑基本清楚；
- 数据层完成了成交量量纲、流通股本短窗口稳定性、换手率回代和收益口径检查；
- comomentum 对波动/偏度的结果为“实现没有完全失效”提供了有用的辅助证据。

但是，审稿结论会被两个问题首先卡住：

1. **安慰剂 p 值的实现与被检验的 pooled 回归统计量不一致。**  `common.py` 中的 `placebo_pvalue()` / `placebo_null_dist()` 按因子逐组估计零分布，而 Table 2、Table 3、Table 4 分位数回归和 Table 7 的观测统计量是合并六因子的 pooled 回归统计量。当前报告的 p 值不是同一统计量下的零分布，因而主表显著性和 11/15 闸门结论必须重算。
2. **两融信号的可得时点没有真正闭合。** 代码使用当月最后交易日的融资余额和当月余额变化，但文档自己承认两融数据为 T+1 披露。若预测从当月末开始，信号是否在当月末收盘时可得没有被证明；这可能构成前视偏差或至少时点定义不严谨。

因此，当前稿件最合适的状态是：

> **研究候选稿，数据工程基本可用；主推断尚未通过审稿闸门。应先修复安慰剂程序、冻结信号时点并全量重跑，再讨论投稿。**

### 当前建议的决策

- **不要直接投稿，也不要以当前 p 值写“系数存活”。**
- **保留研究问题和双测度框架。** 这部分是项目最有价值的资产。
- **将“直接测度”降为“balance-based / more directly observed proxy”，暂时不要写成 holdings-based 或真实拥挤的直接观测。**
- **把主张从“因果效应/机制已证实”收窄为“条件预测关系与测量差异”。**
- **先重建一套同统计量、同置换方案、同多重检验家族的推断结果。**

---

## 1. 已核实的强项：哪些东西可以保留

### 1.1 研究问题和定位有增量潜力

当前最有潜力的贡献不是“在中国也发现拥挤会衰减”，而是：

- 同一市场、同一组风格因子中，比较一个依赖收益相关性的间接代理和一个基于融资余额变化的制度型代理；
- 利用 A 股两融数据的可得性，讨论单一拥挤测度无法识别的**测量层差异**；
- comomentum 不预测收益，但在至少一个波动 horizon 和两个偏度 horizon 上有预测关系，这比单纯报告一个无显著结果更有解释价值。

这使稿件有机会从“单国复制”转向“测量边界条件”。不过，当前标题和部分正文仍没有彻底完成这一定位转换，见第 6 节。

### 1.2 数据工程的几个基础检查是有效的

已检查的结果包括：

- 初始 universe：5,418 只股票，含 257 只退市股；
- 交易日：2,674 天；
- 两融原始文件：沪、深各 2,674 个文件，当前抽查没有空文件，股票日期重复记录为 0；
- 清洗月频面板：480,057 个 stock-month；
- 流通市值推导的成交量量纲、换手率回代和 20 日内流通股本噪声结果较好；
- 因子收益序列在各因子内部没有月份跳跃，`ls_fwd1` 在 `ym=t` 记录、实现于 `t+1` 的设计在代码注释和实际列结构上是一致的；
- LaTeX 可以生成 PDF，引用没有缺失 key，图文件存在。

这些结果支持继续做项目，但它们只证明“数据管道的一部分内部一致”，不能替代 PIT、经济定义和推断有效性的审查。

### 1.3 工程化意识明显优于普通初稿

项目已经有：

- 原始/清洗/报告分层；
- 合成数据冒烟测试；
- `data_dictionary.md`；
- 断点续传、日志和数据质量审计；
- 对真实数据才会暴露的停牌、月末端点、行业名称匹配问题做过回溯修正。

这是项目的明显优点。下一步应把这些经验转化为**可供审稿人复核的 replication package**，而不是只留在中文任务计划的注释中。

---

## 2. P0 阻断问题：不修复不能把主结果当作有效推断

## P0-1：安慰剂零分布与观测统计量不匹配

### 代码证据

- `code/common.py:295–313` 的 `placebo_null_dist()`：
  - 先按 `factor` 分组；
  - 对每个因子单独循环平移拥挤度；
  - 对这个单因子子样本单独调用 `hac_ols()`；
  - 将这些因子内 t 值混在一起形成零分布。
- `code/common.py:316–332` 的 `placebo_pvalue()` 复用同一逻辑。
- `code/09_regressions.py:112–124` 的 Table 2 观测回归却是在六因子合并面板上进行，并含因子固定效应；Table 3 的 ①/③层也使用合并面板。
- `code/10_tail_risk.py:44–73` 的分位数安慰剂同样按因子逐组估计，而观测分位数回归是合并面板加因子固定效应。
- `code/15_volatility.py:110–120` 也使用 pooled 数据的观测回归，再调用相同的逐因子安慰剂。

换言之，当前做的是：

> 用 pooled estimator 得到观测 t，再用 factor-specific estimators 的 t 混合分布作为零分布。

这不是同一个检验统计量的置换分布。因子固定效应、样本量、控制变量的有效秩、残差协方差结构和权重都不同。

### 为什么是硬伤

安慰剂 p 值不是一个独立于估计量的“通用校正数”。它必须回答：

> 在零假设成立时，**同样的样本、同样的设计矩阵、同样的 pooled estimator、同样的标准误算法**会产生怎样的统计量？

当前程序没有做到这一点。因此：

- Table 2 的星号不再具有已声称的含义；
- Table 3 的六个证据单元不能用于闸门判定；
- Table 4 的“中位数显著、左尾不显著”需要重算；
- Table 7 的“comomentum 实现没有失效”仍可能作为描述性方向，但其 p 值也需要重算；
- `gate_2026-11-15_verdict.md` 的“crowd_margin 6/6 存活”不能作为当前版本的最终结论。

### 建议的修复方案

将置换程序分为两个明确接口：

1. **Pooled 主检验：** 每次循环平移都复制完整的 pooled panel，按因子变换测度列，然后对完整 panel 重新估计原来的 pooled 回归，记录一个 pooled t。
2. **逐因子检验：** 只对单个因子估计单因子模型，零分布也只来自该单因子模型。

对于 pooled 面板，至少要明确两种可选零假设：

- **共同滞后：** 六个因子使用同一个 circular shift，保留跨因子共同月份冲击；
- **因子内独立滞后：** 六个因子各自滞后，但每次仍将完整六因子 panel 合并估计。

正文必须说明选择哪一种以及为什么。更稳妥的第一版是共同滞后，因为它不人为破坏跨因子时间结构。

伪代码应接近：

```python
for k in valid_shifts:
    perm = df.copy()
    for factor, idx in df.groupby("factor").groups.items():
        perm.loc[idx, measure] = np.roll(df.loc[idx, measure].to_numpy(), k)
    result = fit_exact_same_pooled_model(perm)
    null_t.append(result[measure].t)
```

实际实现还应：

- 对有限置换使用 `(1 + count) / (1 + number_of_shifts)`；
- 保存每次置换的有效样本量、被丢弃列和收敛状态；
- 用同一函数生成 observed statistic 和 placebo statistic，避免模型规格漂移；
- 修复后重跑合成零数据，并新增一个**跨因子相关、共同市场冲击、不同因子样本起点**的合成测试，而不是只测独立同分布空数据。

**优先级：P0；修复后 Table 2、3、4、7、13 和闸门报告全部重跑。**

---

## P0-2：两融信号的可得时间没有闭合，存在月末前视风险

### 代码和文档证据

- `code/08_crowding.py:72–77` 取每只股票每月最后一个两融余额观测，并计算当月余额减上一个观测余额；
- `code/08_crowding.py:219–232` 的 comomentum 窗口包含直到当月末的日收益；
- `code/08_crowding.py:298–322` 用当月 `d_mb` 形成 `crowd_margin`；
- `paper/main.tex:176–181` 把该变量定义成当月余额变化；
- `input/01-Research-Proposal-因子拥挤与A股因子衰减.md:86–90` 自己写明“两融数据 T+1 披露”。

如果两融明细的日期是交易所数据发生日，但该数据在下一个交易日才公开，那么在月末 t 形成信号时，月末余额未必可知。此时信号应：

- 使用截至最后一个**可得披露日**的余额；或
- 将预测起点整体后移一个披露日/月；或
- 在定义中明确“t 是披露月而不是余额发生月”。

当前代码把“观测日期”和“研究者可使用日期”视为同一日期，尚未提供接口文档或时间戳证据证明这一点。

### 必须补的验证

对至少 3 个代表月份和 3 只股票保存：

1. 余额发生日期；
2. 交易所/接口发布日期或可获得日期；
3. 信号形成日；
4. 首个被预测收益日。

如果公开数据没有单独的披露时间戳，采用保守规则：把余额信号滞后一个交易日，并做滞后一个月的稳健性。正文中画出“信息可得—分组—收益实现”的时间线。

**优先级：P0；在确认前不能使用“无前视偏差”表述。**

---

## P0-3：当前主张把融资余额代理说得过于“直接”

两融融资余额是更接近资金来源的可观测变量，但它不是：

- 账户级真实持仓；
- 因子层面资金权重；
- 纯粹的净买入流；
- 独立于股价、融资偿还、利息、标的扩容和披露制度的拥挤量。

当前稿件中出现了：

- `paper/main.tex:89–94` 的 “holdings-based one”；
- `paper/main.tex:176–181` 的 “direct: margin crowding”；
- `paper/main.tex:291–294` 的 “flow of leveraged capital”；
- 摘要中的 “validate ... against a direct one”。

更准确的名称应为：

> **margin-financing-balance-based crowding proxy** 或 **more directly observed financing proxy**。

同时，`ΔMB / FloatMV` 不是 `fin_buy - fin_repay`。它可能含有余额重分类、标的进入/退出、跨月缺失、融资余额口径变化等成分。建议至少增加：

- `fin_buy - fin_repay` 的流量代理；
- 余额水平；
- 两融覆盖率；
- 仅在连续观测月份上计算的余额变化；
- 多头端、空头端分开结果。

当前“长端 alone 变弱”反而说明“多空差额就是拥挤”的经济解释并不稳，应在正文中作为限制而非轻描淡写的稳健性结果。

---

## 3. 数据层和 PIT 审计：当前可用，但仍有几处未闭合

## 3.1 `check_fetch.py` 的完整性检查实际上没有检查“交易所 × 日期”完整性

`code/check_fetch.py:34–44`：

```python
have_m = {p.stem for ex in exs for p in (MARGIN / ex).rglob("*.parquet")}
```

这会把两个交易所的文件名合并成一个日期集合。若某日只有 SSE 文件而没有 SZSE 文件，集合仍然包含该日期，检查会误报完整。

此外，`03_fetch_margin.py:101–103` 在失败时会将 `None` 规范化成空表并写入文件；`check_fetch.py` 只检查文件名，不检查空表或必需字段，因此失败的 API 请求可能被当成“已完成”。

### 修复

- 使用 `(exchange, date)` 二元键检查 5,348 个单位；
- 对每个文件检查 `date`、`code`、`margin_balance` 是否存在；
- 对允许为空的日期建立明确白名单；
- 对接口失败写入 `.failed.json`，不要写成成功的空 parquet；
- 运行结束报告每个交易所缺失日期、空文件数、股票数分布和异常字段。

当前生产数据抽查显示沪深各 2,674 个文件、没有空文件，这说明**本次落地结果目前看是完整的**；但复现检查器仍有漏洞。

## 3.2 行业剔除使用当前行业快照，不是历史 PIT 行业

`data/clean/industry.parquet` 的 `updateDate` 为 2026-10-12，来源是当前行业分类。`05_build_panel.py:73–74` 将它映射到全部历史日线，再在 `:167–178` 剔除金融行业。

这对“只剔除金融行业”仍可能产生历史误分类：公司重组、上市时行业不同、退市公司分类更新都会影响过去样本。需要：

- 查明 Baostock 行业接口是否提供历史生效日期；
- 如果没有，承认这是 current classification backfill；
- 做“不剔除金融行业”和“以当前分类剔除”的敏感性；
- 报告历史每年被剔除的公司数，而不是只报告当前匹配成功。

当前代码的关键词匹配修复是正确方向，但“真实行业 PIT”尚未被证明。

## 3.3 pbMRQ / peTTM 的 PIT 证据还不够支撑全市场断言

当前证明主要是：用浦发银行 `close/pbMRQ` 反推账面价值台阶，发现变化更接近披露日。这个检查有价值，但不能单独证明：

- 所有股票、所有年份都按同一更新规则；
- peTTM 与 pbMRQ 在除权、停牌、重组和数据缺失时没有回填；
- Baostock 当前字段没有历史修订或数据库回填。

建议把验证扩展到：大盘、金融、制造、创业板/科创板、退市样本，各抽多只股票；对比公告日期；记录异常跳变。正文改成“we verify the PIT interpretation on a multi-stock audit”，除非真的完成多股票验证。

## 3.4 隐含 ROE 校验的结论应降级

`reports/roe_proxy_validation.md` 的结果为：80 只股票、1,970 个配对，Pearson 0.647、Spearman 0.664。这个结果说明有信息量，但仍有：

- 同一股票多期观测造成的非独立性；
- 只在披露日后 30 天取中位数，存在选择窗口；
- `pbMRQ / peTTM` 是 TTM 与最新净资产的混合口径，`roeAvg` 是单季口径；
- QMJ 实际是截面排序，应该报告每月横截面 rank correlation，而非只报告 pooled correlation。

建议报告：按月的 Spearman 分布、股票聚类置信区间、QMJ 分组一致率，并将“隐含 ROE 是可接受代理”写成有限样本下的代理验证，而不是“天然正确”。

---

## 4. 因子构造与组合收益：需要补定义、修口径或承认限制

## 4.1 论文没有给出六个因子的可复现定义

`paper/main.tex` 的 Data and methodology 只说 six style factors，没有把 VAL、SMB、MOM、QMJ、LOWVOL、TURN 的公式、排序方向、缺失处理和形成窗口写入正文。审稿人不能仅凭表格复现核心资产定价变量。

至少应补一张“Variable construction”表：

| 因子 | 代码定义 | 排序方向 | 形成窗口 | 缺失/异常处理 |
|---|---|---|---|---|
| VAL | `1/pbMRQ` | 高值做多 | 月末 | pb≤0 剔除 |
| SMB | `-log(float_mv)` | 小市值做多 | 月末 | 流通股本推导 |
| MOM | 前 12 个月、跳过最近 1 个月复利收益 | 高值做多 | 完整自然月 | 缺月为 NaN |
| QMJ | `pbMRQ/peTTM` | 高值做多 | PIT 日频估值 | pe/pb 正值 |
| LOWVOL | `-std(ret, 60)` | 低波做多 | 至少 40 日 | 当前代码不是严格 60 日 |
| TURN | `-mean(turn, 20)` | 低换手做多 | 至少 15 日 | 当前代码不是严格 20 日 |

## 4.2 论文写“60/20 日窗口”，代码实际允许 40/15 日

`code/06_build_factors.py:53–54`：

```python
rolling(60, min_periods=40)
rolling(20, min_periods=15)
```

这不是小的文字差异。短历史股票可能以较短窗口进入极端分位，且缺失结构与股票上市年龄、停牌和制度板块相关。需要：

- 要么改为严格 60/20；
- 要么正文明确“窗口长度为 60/20、最少有效观测为 40/15”，并做严格窗口稳健性；
- 报告每月实际有效窗口长度。

## 4.3 缺失收益被填成 0 的经济含义需要说明

`code/06_build_factors.py:60` 使用 `p[

对于经过 `tradestatus==1` 筛选后的正常交易日，缺失收益理论上应很少；但把未知收益直接当作 0 会把数据缺失变成“无价格变化”。应在构造前统计缺失率，并做两种版本：

- 严格版本：缺失日使该股票该月收益无效；
- 当前版本：缺失日填 0。

如果两者结果不同，主结果应采用严格版本，当前版本作为敏感性分析。

### 4.4 “停牌超过 10 日”规则并不等于可交易规则

代码使用 `n_days >= 10`，但 `n_days` 是清洗后仍有交易记录的天数。它不是完整的可交易状态变量，也没有处理涨跌停、T+1、无法卖空和部分成交。因此正文不要说“避免不可交易”，应说“排除交易记录过少的 stock-month”。若因子收益被解释为资产定价组合而非可执行策略，这一限制可以接受，但必须明确。

### 4.5 每只股票使用自己的月末快照，造成非同步形成日

`06_build_factors.py:66–68` 选择每只股票当月最后一条记录；停牌股的最后记录可能早于交易所月末。`08_crowding.py` 又用交易日历的统一月末作为 comomentum 窗口端点。结果是分组暴露、comomentum 窗口和 margin balance 可能不是同一个信息时点。建议做严格版本：只纳入统一月末仍有观测的股票，或在个股层面记录最后可得日期并把信号整体滞后。至少报告最后观测日距统一月末 0、1–3、4–10 个交易日的分布。

### 4.6 comomentum 实现包含随机抽样，应报告 Monte Carlo 稳定性

`08_crowding.py:33–40,245–251` 每个多空端最多随机抽取 100 只股票，固定种子 `20261016`。这保证复现，却不能证明测度稳定。应增加全量/50/100/200 股票版本、至少 20 个随机种子的均值与标准差、月度 rank correlation，以及不同 `MIN_PAIRS` 的敏感性。若全量计算不可行，论文应称为 sampled pairwise-correlation proxy，并把抽样误差列为限制。

### 4.7 两两完整相关会使不同 pair 使用不同有效窗口

`DataFrame.corr(min_periods=60)` 比整行 `dropna` 合理，但不同 pair 使用不同有效日期，平均相关系数可能随停牌结构变化。应报告每月有效 pair 数、有效观测数分布、共同完整窗口版本和按共同观测数加权版本。

### 4.8 FF3 不能直接称为标准 FF3

`08_crowding.py:82–114` 的 MKT、SMB、HML 是在本项目清洗样本上按十分位构造的内部基准，不是标准数据库因子。正文应改为 `three-factor benchmark constructed from the same cleaned universe`，并给出市场腿、规模腿、价值腿、权重和再平衡规则。最好增加外部标准因子或替代市场因子版本。

---

## 5. 主回归与统计推断：除 P0 外的关键问题

### 5.1 `R_{t→t+h}` 用简单收益相加，不是累计复合收益

`09_regressions.py:56–57` 对连续 `ls_fwd1` 直接 rolling sum，但论文写的是 cumulative long–short return。主结果应二选一：改代码为 `(1+r).rolling(h).apply(prod)-1` 并重跑，或把正文和表头改成 cumulative simple return，并解释这是预测回归的 horizon sum。功效分析的年化换算也要与该定义一致。

### 5.2 重叠 horizon 的回归需要比 HAC(6) 更严格的说明

h=12 的被解释变量重叠 12 个月，且信号本身有 12 个月形成窗；`NW_LAGS=6` 是配置硬编码值，缺少依据。应让 lag 覆盖预测重叠结构，或把 month-block/circular inference 作为主推断；不同 horizon 应有各自的 null distribution。

### 5.3 合成校准证据不能写得过强

三个 synthetic draws、96 个 p 值的 KS 检验只能说明没有检测到明显偏离，不能证明方法正确。建议写成 `we did not reject uniformity in a limited synthetic calibration`，并公开种子、DGP、跨因子相关结构、缺失结构和全部 p 值。

### 5.4 多重检验没有真正处理

当前同时搜索 2 个测度、4 个 horizon、4 个规格、正交化层、6 个因子、分位数、波动/偏度、子样本和窗口。`≥2` 个单元 p<0.10 可以作为预先闸门规则，但不能当作 family-wise 显著性。应冻结 primary family，主结果采用 stepdown/FDR 或明确其余结果为 exploratory，并报告家族内最小 p 与校正结果。

### 5.5 “控制后系数变强”不能直接解释为纯拥挤被识别

加入过去收益后 margin 系数变强，可能来自遗漏变量修正、坏控制、共线性或测量误差变化。正文应写 `incremental predictive association conditional on past factor returns`，不要写“剩下的才是纯拥挤”。需要报告第一阶段相关、VIF/条件数和置信区间。

### 5.6 正交化是全样本变换，不应称为实时纯净信号

`09_regressions.py:146–161` 在完整样本上对 crowding 做 first-stage residualization。作为样本内 FWL 变换可以用于条件预测识别，但不是实时交易信号。正文应明确这一点；若要主张可实时构造，应做 expanding/rolling first stage。

### 5.7 共同月份冲击仍未充分处理

六个因子共享月份内市场冲击。六个因子聚类确实太少，但 pooled NW 也不自动处理跨因子相关。应增加 month-block bootstrap、Driscoll–Kraay 或谨慎的二维/wild bootstrap 作为对照。

---

## 6. 结果解释、机制和否定结论

### 6.1 comomentum 预测波动不是“测度没有误测”的充分证明

它最多说明实现产生了与既有文献相符的二阶矩模式；残差相关也可能反映行业、主题、流动性、停牌或共同信息冲击。正文应把结论降为 qualitative consistency，并增加行业中性残差、流动性/市场波动控制或替代因子基准。

### 6.2 相关系数低不等于两个测度已经被证明捕捉不同经济对象

Table 1 的低相关也可能来自不同量纲、非线性关系、月末错位、覆盖率变化和 comomentum 抽样噪声。建议写成 `limited linear comovement in this sample`，而不是“两个测度捕捉的是不同东西”。可补 distance correlation 或按覆盖率分层。

### 6.3 功效分析不能把 null 结论写成强否定

MDE 依赖当前模型、当前 placebo critical value 和 bootstrap DGP，并且主规格 C 的功效不能代表所有规格。更稳妥的表述是：`the null is informative relative to effects above the estimated detection threshold in this design`。最好补外部文献效应量映射、置信区间或等价性检验。

### 6.4 尾部结果目前只能是辅助结果

`10_tail_risk.py` 没有正式检验 q=0.10 与 q=0.50 系数差异，左尾不显著也不能证明没有 crash risk；crash threshold 使用全样本历史 5% 分位数，含有未来样本信息。需改为形成时点可得阈值、系数差异检验和正式 block/permutation inference。当前不要把“中段而非左尾”放进摘要级贡献。

---

## 7. 结果—论文数字与表图一致性

主要 CSV 与生成的 LaTeX 表数值总体相符，Table 2 星号确实由 `p_placebo` 生成，这是优点。但数值一致不等于统计量有效，P0-1 仍优先。

论文与代码需要逐项统一：

1. 正文 60/20 日窗口与代码 40/15 最小有效观测不一致；
2. `direct/holdings-based` 应改为融资余额代理；
3. `cumulative return` 与代码 horizon sum 不一致；
4. “full A-share cross-section”应说明当前行业快照和 `ret_next` 筛选限制；
5. “free of survivorship bias”应限定为初始 universe 含退市股，而不是保证最终因子面板没有任何生存性选择；
6. 双轴图适合展示量纲，不适合直接支撑相关性，建议加标准化同轴图或散点图。

格式检查结果：

- `check_format.py --strict`：FAIL，30 个问题；
- `check_tables.py`：多列小数位不一致，星号无表注；
- LaTeX 编译虽成功，但多张表存在严重 overfull hbox，Table 5 约 187.8pt；
- Table 3 使用 ①②③，Latin Modern 字体报告多次 Missing character；
- 当前 PDF 为 15 页，而项目记录写 13 页，说明版本没有完全同步；
- `verify_bib.py` 报 `lee2026returndecay` 未使用，另有 DOI/页码建议；
- Acknowledgements、利益冲突、资助和数据可得性仍未完成。

每个带星号的表都应加：

> Notes: Stars denote circular-shift permutation p-values: * p<0.10, ** p<0.05, *** p<0.01. HAC t-statistics are shown for comparison. The permutation unit, number of valid shifts, fixed effects, and horizon-return definition are reported in the table or appendix.

---

## 8. 参考文献和论文定位

### 8.1 参考文献太少，且有未使用条目

`references.bib` 共 10 条，正文使用 9 条。若确实用于“推断方式敏感”的讨论，应准确引用 `lee2026returndecay`；否则删除。建议补充并核实中国两融制度、标的扩容、因子动量、factor decay、A 股因子构造和 comomentum 二阶矩定义的文献。每条应有 DOI、卷期页码或稳定 URL。

### 8.2 “不是 replication”不能只靠一句话解决 scope 风险

当前标题仍将中国市场置于第一层。更好的标题方向是：

- `When Return-Correlation Crowding Meets Observable Leverage`
- `Measuring Factor Crowding When Leverage Is Observable`
- `Return-Correlation and Margin-Balance Proxies for Factor Crowding`

A-share evidence 放入副标题或摘要，核心卖点写成 measurement comparison。

### 8.3 降级机制句子

需要数据或引用支撑的句子包括：`financing is overwhelmingly retail`、`long side dominates short side by an order of magnitude`、`the flow of leveraged capital` 和 `the inference result carries beyond both`。当前设计最多支持条件预测关系，不支持这些强机制表述。

---

## 9. 论文结构和语言

### 9.1 引言有研究日志痕迹

`main.tex:96–125` 的 first/second/third 叙述过多解释内部开发过程和 synthetic panel。FRL 短文应压成：数据/信号定义、预先指定的识别风险、校准和主结果。详细修错过程放 replication note 或 appendix。

### 9.2 结论缺少明确限制段

需要明确：非因果、融资余额代理不是账户级持仓、行业当前快照、交易可执行性不在范围、未来需要账户级数据或标的扩容 DID。当前结论不能只重复结果和外推。

### 9.3 否定结论需 evidence-bounded

把“没有 predictive content”改成：

> `we do not find a robust return-predictive association for comomentum in this sample and design.`

特别是在推断程序修复前，不得使用绝对否定。

---

## 10. 代码与可复现性工程

### 10.1 流水线失败不阻断

`run_all.sh:31–41` 返回 rc 后主流程没有逐阶段检查；阶段失败后仍可能写 `data/DONE`。应为每阶段生成含输入/输出 hash、时间和 rc 的 manifest，非零 rc 默认停止，`data/DONE` 只在全部成功且完整性校验通过后写入。

### 10.2 当前工作目录不是 Git 仓库

没有 commit hash，无法准确锚定论文 PDF、CSV 与代码版本。建议初始化 Git，排除大原始数据或用 manifest/DVC 管理，提交代码、配置、论文源和报告，并把最终 commit hash 写入运行报告。

### 10.3 绝对路径过多

多个脚本硬编码 `/home/user/金融科研1014`，`13_power.py` 直接用绝对路径载入 `09_regressions.py`。统一使用 `FIN_ROOT`/`--root`、配置相对路径和 `Path(__file__)`，让复现包可移植。

### 10.4 结果版本漂移

应为 analysis CSV、publication CSV、LaTeX table、PDF 统一 run ID。当前 PDF 15 页与项目记录 13 页不一致，是版本漂移的直接信号。

### 10.5 端到端 fixture

除现有合成测试外，增加固定小型 fixture，验证 pooled null、已知 beta 的功效、`ls_fwd1` 时点、停牌/缺失/扩容处理、CSV—LaTeX 数字一致和一键 PDF 构建。

---

## 11. 逐阶段改进清单

### Stage 0：冻结版本与研究问题

- [ ] 建 Git 和 `run_manifest.json`，记录数据/配置/代码 hash；
- [ ] 冻结 primary estimand：信号、起点、h、主规格和主测度；
- [ ] 将 direct 改名为 margin-financing-balance proxy；
- [ ] 明确非因果、非交易策略、非账户持仓识别。

### Stage 1：信息时点和数据完整性

- [ ] 核实两融披露时点，必要时滞后信号；
- [ ] 修复 `(exchange,date)` 完整性检查；
- [ ] 检查空 parquet、必需列和失败文件；
- [ ] 添加 `available_date`；
- [ ] 报告标的扩容、首次观测和连续观测覆盖；
- [ ] 做当前行业快照与不剔除金融敏感性。

### Stage 2：推断程序

- [ ] 实现 pooled estimator 的 exact permutation null；
- [ ] observed 和 placebo 复用同一 fit 函数；
- [ ] 使用随机化 p 值加一修正；
- [ ] 不同 horizon 分别校准；
- [ ] 增加 month-block bootstrap；
- [ ] 处理主检验家族的多重比较；
- [ ] 重跑 Table 2、3、4、5、6、7 和 gate。

### Stage 3：因子与 comomentum

- [ ] 严格 60/20 与当前 40/15 并列；
- [ ] 缺失收益严格版与 fill-zero 版并列；
- [ ] 统一月末形成日与最后可得日并列；
- [ ] comomentum 做抽样和 seed 稳定性；
- [ ] pairwise complete 与 common-window 并列；
- [ ] 内部 FF3 与外部替代因子并列；
- [ ] 加行业、流动性和市场波动控制。

### Stage 4：主表与主张

- [ ] 主表只保留预先指定测度、规格和 horizon；
- [ ] 其余结果标 exploratory；
- [ ] 表中补 beta、SE、NW t、exact permutation p、N、R²、固定效应和控制变量；
- [ ] 复合收益与简单 horizon sum 做敏感性；
- [ ] 尾部结果完成正式系数差异检验后再升级。

### Stage 5：论文与格式

- [ ] 标题突出 measurement comparison；
- [ ] 加 variable construction 表和 timing figure；
- [ ] 加 limitations/future work；
- [ ] 修复 overfull hbox 和 Unicode 缺字；
- [ ] 用 Layer 1/2/3 代替 ①②③；
- [ ] 加完整星号表注，统一小数位；
- [ ] 完成 COI、funding、data/code availability 和 acknowledgements；
- [ ] 以投稿时官方 Guide 重新核验字数、摘要和格式。

---

## 12. 推荐的新版主表结构

### Table 1：样本、变量和测度描述

股票数、stock-month、每月有效因子数、两个 proxy、覆盖率、首次观测比例、均值、标准差和分位数。

### Table 2：预先指定的主预测回归

只保留两个 proxy、h=1/3/6、一套主控制（factor FE、past returns、margin level、coverage），报告 beta、SE、NW t、exact permutation p、N 和 R²。

### Table 3：测量对照

在同一 pooled 模型和同一置换程序下并列收益、波动和偏度结果，比当前把辅助波动表后置更清楚。

### Table 4：稳健性

严格窗口、滞后信号、严格缺失、等权、6/24 月窗口、行业/流动性控制。

其余因子看板、quantile、功效、synthetic calibration、数据字典和 PIT 证据放 appendix/replication package。

---

## 13. 审稿式最终结论

**Recommendation: Major revision before submission.**

本文提出了有潜在价值的测量问题：在融资余额可观测、套利约束不同的市场中，收益相关性型 comomentum 与融资余额型拥挤代理是否给出一致的收益预测。数据覆盖和工程记录较好，结果也具有一定非对称性。然而，当前版本的主推断尚未可靠：置换零分布与 pooled 观测统计量不匹配，融资余额信号的披露时点未闭合，且若干因子、累计收益和可复现定义与代码不一致。稿件还需要显著收窄 direct/holdings-based/机制已证实等表述。

修复 exact pooled permutation、确认两融可得时点、重跑全部表格并增加时点/窗口/缺失/抽样/共同冲击稳健性之后，论文可以重新评估。即使重跑后两融系数减弱，研究问题仍然成立；如果效应只在部分规格或子样本存在，应诚实改写为条件性预测关系。

### 当前最强可保留句子

> When financing balances are observable, a return-correlation proxy and a financing-balance proxy need not be treated as interchangeable measures of factor crowding.

### 当前应删除或降级的句子

> The direct flow of leveraged capital predicts factor decay, while comomentum does not, proving that the two measures capture different things.

更稳妥的版本是：

> In the current A-share sample, the margin-financing-balance proxy shows a conditional negative association with subsequent factor returns, whereas the return-correlation proxy does not. The two proxies have limited linear comovement, but the evidence does not by itself identify their underlying economic objects.

---

## 14. 最小执行闭环

### 第一轮：先修推断，不改故事

1. 建 Git 和 run manifest；
2. 修 exact pooled permutation；
3. 修 `check_fetch.py`；
4. 核实两融 availability date；
5. 以滞后信号重跑 Table 2；
6. 只看主规格 h=1/3/6；
7. 根据修复后的结果决定是否保留当前主张。

### 第二轮：修数据和变量定义

1. 严格窗口；
2. 缺失收益严格版；
3. 统一月末形成日；
4. comomentum 抽样稳定性；
5. 内部 FF3/替代 FF3；
6. 行业敏感性；
7. 复合 horizon return。

### 第三轮：重写论文

1. 标题和摘要改成测量对照；
2. 加 variable/timing 表；
3. 重做主表和表注；
4. 将尾部、功效和 synthetic calibration 放附录或辅助结果；
5. 加限制和数据可得性声明；
6. 修复所有 LaTeX overfull、缺字、星号注释和小数位。

### 投稿门槛

只有满足下面全部条件，才建议进入投稿信和期刊格式阶段：

- [ ] pooled permutation 与 observed estimator 完全一致；
- [ ] 两融 availability date 已证明或采用保守滞后；
- [ ] 主结果在至少一种滞后时点下仍方向一致；
- [ ] 主结果不完全依赖单一窗口/缺失实现；
- [ ] primary estimand 和多重检验家族已冻结；
- [ ] 所有表的 p 值、星号和文字一致；
- [ ] 论文明确不做因果识别、不做交易策略、不等同账户级持仓；
- [ ] 代码、数据字典、manifest、commit hash 和一键复现命令齐全；
- [ ] COI、funding、data/code availability 完成；
- [ ] LaTeX 严格检查不再 FAIL，PDF 无 overfull hbox 和 missing character。

---

**报告完**
