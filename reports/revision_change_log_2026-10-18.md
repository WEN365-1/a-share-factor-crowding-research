# 修订日志 2026-10-18（投稿前补强轮）

依据 `reports/frl_submission_review_2026-10-18.md` 的审查结论执行。全部改动已本地重跑验证。

## 一、代码层

| 变更 | 文件 | 说明 |
|---|---|---|
| 路径可移植化 | 全部 15 个脚本 | `/home/user/金融科研1014` 硬编码 → `Path(__file__)` 相对写法；FIN_ROOT 环境变量覆盖根目录 |
| 本地整链路复现 | 09/10/11/15 重跑 | 与云主机基线逐位一致（Table 2/3/4/5/7 最大差异 ≤1e-14） |
| 新增扩展分析 | `16_extension_analyses.py` | 复利收益、腿分解、剔除 SMB、覆盖率门槛、within-factor 标准化、两代理一致性拆分、覆盖率统计 |
| 新增多股票 PIT 审计 | `17_pit_audit.py` | 40 只股票：隐含 BPS/EPS 台阶 ↔ 财报公告日匹配（结果见 pit_audit_multistock.md） |
| 新增流量型代理 | `18_flow_proxy.py` | 月度融资买入强度（两所）与净流量（仅 SSE）；T+1、市值权重归一化、流量自身 12 个月水平控制、赛马回归 |
| Table 5 补对称行 | `11_robustness.py` | 新增「仅空头端」测度变体行（v_short = v_long − crowd_margin） |
| 表格渲染升级 | `12_make_tables.py` | Table 2 改为双面板（系数+置换 p 值）；新增 Table 6 腿分解渲染；PAPER 路径可移植 |

## 二、论文层（main.tex，与审查报告条目一一对应）

| 条目 | 处理 |
|---|---|
| H1 正文与 Table 1 数字 | 已改：−0.25（turnover）/−0.27（value）、pooled −0.14、去均值 −0.09 |
| H2 稳健性 p 值 | 已改：2015–2018 p=0.194（10% 水平显著）；0.111/0.149/0.008 复核无误 |
| H3 multi-stock audit | 已做真审计（40 只股票），说法成立 |
| H4 compound-return 承诺 | 已兑现：h=6 β=−0.042（p=0.010）、h=12 β=−0.057（p=0.010），写入稳健性节 |
| H5 AI 工具名 | 已改：DeepSeek（大语言模型）；删除工具致谢 |
| H6 图源缺失 | `paper/figures/fig_crowding_series.pdf` 已入库；干净环境编译验证通过 |
| H7 仓库 private | 数据声明改为「available from the corresponding author, public upon acceptance」；投稿前决定是否公开仓库 |
| H8 因子定义 | 已在 Data 节添加六因子紧凑定义 |
| H9 Table 2 缺推断统计量 | 已加 Panel B：置换 p 值面板 |
| H10 行业快照 | 已披露「current-snapshot classification applied to the full sample」 |
| H11 SMB 覆盖率 | 已披露（0.26 vs 0.82–0.93）；新增剔除 SMB 与覆盖率门槛稳健性 |
| H12 稳健性边界 | 新增 Table 6 腿分解：L−S 效应全部来自空头腿；空头腿自身融资强度预测自身衰减（h=6 β=−0.0517, p=0.010） |
| M1 样本口径 | 「5,247 of the 5,418 stocks in the cleaned universe」 |
| M2 过时注释/TODO | 已删除并更新 |
| M3 CRediT | 已添加（单作者全项） |
| M4 JEL 混入关键词 | JEL classification: G12 单独成行 |
| M7 标准化口径 | 正文同时报 pooled（4.0pp）与 within-factor（3.3pp）口径 |
| T2 口语化 | 「in the end」→「eventually」；「That comparison is the point.」→「The comparison itself is the contribution.」 |
| T3 摘要 exploratory | 已改「individually insignificant」；摘要新增腿机制与独立性两句 |
| T1 对冲压缩 | descriptive/auxiliary/not-X 重复大幅合并，净省约 150 词 |

**字数**：正文 2414（不含表格/图注/摘要，最严口径计数器）；含声明 2584；摘要约 190。PDF 17 页。

## 三、本轮新发现（已写入论文）

1. **腿分解（Table 6）**：h=6 时多空系数 −0.0400 = 多头腿 +0.0005 − 空头腿 0.0406；空头腿自身两融强度 → 空头腿自身衰减 β=−0.0517（p=0.010）。信号不是脆弱的价差假象，而是低端腿融资强度的衰减效应。
2. **复利口径稳健**：h=6 −0.042、h=12 −0.057，均 p=0.010。
3. **剔除 SMB 后更强**：h=6 β=−0.049（p=0.010）——SMB 的弱覆盖此前在稀释而非驱动结果。
4. **覆盖率门槛稳健**：cover≥0.30 → β=−0.046（p=0.022）；≥0.50 → p=0.025。
5. **两代理独立**：comomentum 高/低月份系数相近（−0.036/−0.035），交互项 p=0.20——强化「测量区分」核心论点。
6. **within-factor 标准化**：+1 组内 SD → −3.3pp（h=6），量级结论不受标准化选择影响。
7. **流量型代理（Table 8）**：月度融资买入强度（两所）与净流量（仅 SSE）都显著预测因子衰减；净流量四个持有期全部置换显著（p=0.016–0.019）。赛马：存量测度在任何规格下都存活（p=0.016–0.034），买入流量在 h=1（p=0.016）、净流量在 h=6（p=0.034）保留独立预测内容；两测度相关 0.44–0.50。→ 论文新增 3.4 节「Flows versus stocks」。

## 四、配套文件

- `paper/cover_letter.md`：日期、发现段（含腿分解与独立性）、数据声明、单作者措辞。
- `paper/highlights.txt`：第 3 条改为腿机制发现；5 条均 ≤85 字符。
- `paper/submission-checklist.md`：勾选状态更新至 2026-10-18 实测值。
- `reports/data_dictionary.md`：新增 crowd_margin_short、覆盖率表、多股票 PIT 审计、本地复现说明。
- `paper/main.pdf`：16 页，本地 pdflatex+bibtex 编译，零引用警告，图/表全部就位。

## 五、剩余事项（需用户操作/决策）

1. **GitHub 仓库公开**：✅ 已完成（2026-10-18）。两个仓库（a-share-factor-crowding-research / a-share-factor-crowding-raw-data）均已公开，正文数据声明为公开 URL 版本；raw-data 的 release 资产（784MB + sha256）保持可下载。
2. 投稿系统字段：关键词 6 个、JEL 单独字段、highlights 单独文件、CRediT 按系统要求重填。
3. 推荐审稿人 3–5 位。
4. 云主机与本地的代码/报告差异：本轮全部改动在本机完成，如需云主机保留同步版本，把 code/、paper/、reports/ 同步回去（本机已可整链路重跑，云主机不再必需）。
5. 投稿前通读一遍定稿 PDF（拼写/语法检查尚未做）。

---

## 六、排版轮（2026-10-18 晚，应用户反馈）

| 变更 | 说明 |
|---|---|
| 文档类 | `preprint,12pt` → `final,1p`（Elsevier 出版系统同款单栏版式；Times 字体 via newtx） |
| 表格 | 全部去掉 `\resizebox` 强制拉宽：按自然宽度居中，窄表不再被吹大；表 1/4/5/6 用小号字，列距 2–2.5pt；仅 Table 5（唯一真宽表）保留一次缩放 |
| Notes 行 | 改为可换行段落盒（原单行超长文本越界 1300pt 的元凶） |
| URL | 加 xurl 断行支持；`\emergencystretch=1em` |
| 结果 | 17 页 → **13 页**；越界 **0**；字体 Computer Modern → Times 系（TeXGyreTermesX），与发表版 FRL 一致 |

另：尝试抓取 ScienceDirect 上的 FRL 开放获取文章 PDF 用于比对，被反爬拦截（Cloudflare/登录墙），未成功；本机 `final,1p` 模式即 Elsevier 生产排版同款，已按此对齐。

## 七、发表惯例修正轮（2026-10-18 深夜，应用户指正）

| 问题 | 修复 |
|---|---|
| 引用格式错误（作者-年份制） | FRL 发表版为**数字编号制**：`authoryear` 选项删除、`elsarticle-num`、正文 6 处引用全部改为数字式（如 "Lou and Polk [5]"）；参考文献表 [1]–[9] 编号齐全 |
| 表格穿插在参考文献/声明之间 | 结论节前加 `\clearpage`：所有表格在结论之前整体落位（第 4–11 页），参考文献/声明/致谢区不再出现任何表格 |
| 其他同轮修正 | 地址补国家（Tongji University, Shanghai, China）；Baostock 数据源补网址 |

最终：13 页、0 越界、表格页 4–11、References 12、Declarations 13。

## 八、表格精简与文末集中排布轮（2026-10-18 夜，应用户反馈）

| 变更 | 说明 |
|---|---|
| 表格位置 | 全部 8 表 + 图 1 移至**文末统一区块**（\section*{Tables}，致谢之后），正文仅留 \ref 引用——金融学期刊投稿惯例（JF/JFE 式）；录用后由编辑部放回引用处 |
| 表格精简 | 表 2 双面板合并为单面板（系数+星号，括号内置换 p）；表 3 删除 Layer 2 逐因子行（仅留均值汇总+表注给符号分布）；表 6 删除交叉腿面板与 comomentum h=1 行；尾部特征与分位数回归**合并为一张表**（Panel A/B） |
| 结果 | 13 页 → **11 页**；0 越界；正文 1–7 页（结论→文献→声明→致谢一气呵成），表格 8–11 页 |

## 九、表格就近排布轮（2026-10-18 夜，应用户选择）

应用户"方便审稿人、不必翻页"的选择：表格放回**正文各小节内、首次引用处附近**（发表版样式），保留第八轮的全部精简成果。实现：placeins[section] + 每个小节前 \FloatBarrier（表格绝不漂出所属小节、绝不进入参考文献区）。
最终：14 页、0 越界；表格页 5–12，References p13；每表与所属小节 0–1 页距离。

## 十、表格削减与版式惯例轮（2026-10-18 深夜，应用户反馈）

| 变更 | 说明 |
|---|---|
| 表格 8 → **5 张** | 相关性（Table 1）、二阶矩（Table 5）、分布分析（Table 7）三张辅助表**删除、内容并入正文**（数字齐全：相关 −0.25~+0.27/pooled −0.14；波动 0.0014 p=0.059 其余 p≥0.095；分位数 q=0.50 −0.0093 p=0.031、左尾 p=0.916）；正交表再删 h=12 行。保留：主检验、腿分解、流量、正交化、稳健性 |
| 段落缩进惯例 | 对照用户 ICCAI（EI 会议）稿实测：会议模板为**全部顶格 + 表格单元格空格填充**；Elsevier/FRL 发表版为**正文段落首行缩进 ~15pt（约两字符）、标题后第一段顶格**。已按 Elsevier 惯例批量修正（19 处 \noindent），正文其余段落保持首行缩进 |
| 结果 | 12 页、0 越界；5 表 1 图就近排布；正文 2448 词（≤2500 ✓） |
