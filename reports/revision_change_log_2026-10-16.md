# 修复与重跑变更日志

**项目：** Factor Crowding and Factor Decay in China's A-Share Market 研究项目  
**执行环境：** ClawsGO Agent / ClawsGO Science Harness  
**更新日期：** 2026-10-16（北京时间）

## 1. 已实施的代码修复

### 推断程序

- 将 circular-shift permutation null 改为完整 pooled panel 上的同一 `hac_ols` estimator。
- 每次置换在各因子内部平移测度，但保留完整 pooled 设计矩阵、因子固定效应、控制变量和样本。
- 置换位移使用所有因子共同支持的位移范围。
- 有限置换 p 值改为加一修正：`(1 + exceedances) / (1 + B)`。
- Table 2、Table 3、稳健性、波动/偏度和功效分析统一使用修复后的 pooled inference；分位数回归也改为 pooled permutation。

### 两融信息时点

- 原始两融日期视为发生日。
- 保守设定为下一交易日可得，生成 `available_date`。
- 月末信号只使用 `available_date <= signal_date` 的余额快照。
- 增加 PIT 审计日志。最终重跑结果显示：100.0% 的可得日期不晚于信号日，信号日后可得记录为 0.0%。
- 抓取失败不再写入成功的空 parquet；改写 `.failed.json` 标记。
- `check_fetch.py` 改为检查 `(exchange, date)` 二元单位、空文件和必需字段。

### 因子和数据处理

- LOWVOL 和 TURN 改为严格 60/20 有效交易日窗口。
- 月度收益不再把缺失日静默填成 0；月内收益不完整时记为缺失。
- 日频因子收益统一使用 `ls_fwd1` 字段，避免旧 `ls_ret` 时点歧义。
- 6/24 月两融变化稳健性改为按自然月匹配滞后余额，避免把不连续观测当成相邻月份。

### 复现流水线

- `run_all.sh` 改为 `set -Eeuo pipefail`。
- 任一阶段非零退出时停止后续阶段。
- 抓取达到最大轮次仍不完整时停止，不再继续生成下游结果。
- 增加配置路径和 SHA-256 工具接口，支持 `FIN_ROOT` fixture。

## 2. 真实数据重跑结果

最终全量重跑顺序：

```bash
python3 code/08_crowding.py
python3 code/09_regressions.py
python3 code/10_tail_risk.py
python3 code/11_robustness.py
python3 code/15_volatility.py
python3 code/13_power.py
python3 code/12_make_tables.py
```

并重新运行：

```bash
python3 code/06_build_factors.py
python3 code/15_volatility.py
python3 code/12_make_tables.py
```

关键输出：

- 拥挤度面板：777 行，`(ym, factor)` 重复数为 0。
- 主回归面板：约 771 个有效 pooled 观测。
- `table2_main.csv`：32 个主规格单元，键重复数为 0。
- `factor_daily.parquet`：15,622 行，字段为 `ls_fwd1`，缺失数为 0。
- 论文 PDF：成功生成，最终 14 页。

### 修复后闸门

证据单元为 Layer 1 / Layer 3 × `h ∈ {1, 3, 6}`，显著性以 pooled finite-permutation p < 0.10 为准：

- `comomentum`：0/6 显著。
- `crowd_margin`：6/6 显著，系数全部为负。
- `crowd_margin` 闸门判定为“系数存活”，但该结果是条件预测关联，不是因果效应或账户级持仓证据。

代表性修复后主结果：

- `crowd_margin` Layer 1：h=1/3/6 的系数约为 -0.0150 / -0.0260 / -0.0392，permutation p 约为 0.009 / 0.010 / 0.010。
- `crowd_margin` Layer 3：h=1/3/6 的系数约为 -0.0150 / -0.0264 / -0.0383，permutation p 约为 0.009 / 0.010 / 0.010。
- `comomentum` Layer 1/3：h=1/3/6 的 permutation p 均明显高于 0.10。

### 稳健性边界

- 2015--2018：margin proxy p 约 0.33。
- 2019--2021：p 约 0.11。
- 2022--2025：p 约 0.15。
- 等权：p 约 0.008。
- 6 月余额变化：p 约 0.185。
- 24 月余额变化：p 约 0.252。
- 仅多头端：p 约 0.496。

因此正文已改为“条件性预测关联”和“稳健性有限”，没有再写成所有子样本和窗口均通过。

### 第二矩诊断

修复后的 pooled permutation 下，comomentum 的一月实现波动 p 约 0.059；其余波动/偏度单元没有形成统一的强显著模式。因此正文只保留为方向性诊断，不再写成完整复现 Lou--Polk 第二矩结果。

## 3. 已同步的论文修改

- 标题改为 `Measuring Factor Crowding When Leverage Is Observable: Evidence from China's A-Share Market`。
- 将 `direct`、`holdings-based`、`flow of leveraged capital` 等强表述改为 `margin-financing-balance proxy` 或 `balance-based proxy`。
- 明确两融代理不是账户级持仓、纯融资流或因果变量。
- 明确原始日期、`available_date`、共同月末信号和下一月收益的时间线。
- 明确 horizon return 当前为 horizon sum，并将复合收益留作敏感性，而不再称为累计复合收益。
- 删除依赖旧结果的具体 p 值和过强的尾部、波动、机制表述。
- 添加 limitations、future work、funding/COI/data availability 说明。
- 表格标签改为 margin-balance proxy，Layer 1/2/3 不再使用 Unicode 圆圈数字。
- 表格增加 finite circular-shift permutation p 值注释，并用 `resizebox` 限制页面宽度。
- 补充部分参考文献 DOI，并引用 China 单因子 crowding 研究以消除未使用条目警告。

## 4. 仍然存在的投稿前事项

- LaTeX 已成功编译，但仍有少量正文 overfull hbox、hyperref PDF-string warning 和 xeCJK CJK monofont warning，需要在最终投稿模板中继续清理。
- 格式脚本仍报告破折号和命令空格问题；这些不阻止 PDF 生成，但投稿前应逐项清理。
- 表格脚本仍报告其对嵌入式输入表格的小数位/表注识别警告，实际生成表格已有统一 Notes；应再做一次脚本适配或人工核验。
- `lee2026returndecay` 未使用条目已删除，参考文献检查已 PASS。
- 仍未建立 Git commit hash 和完整 run manifest；应在投稿前冻结代码、配置、报告与 PDF 的版本锚点。
- 历史行业分类仍是当前快照回填；正文已承认这一限制。
- 多重检验家族、严格复合收益敏感性和 comomentum 多 seed 稳定性仍应作为最终附录或 replication package 项目。

## 5. 当前建议结论

当前项目已从“主推断存在硬伤、不可直接投稿”提升到“核心推断和两融 PIT 口径已修复、真实数据已重跑、研究候选稿”。

推荐主张：

> In the current A-share sample, the margin-financing-balance proxy shows a conditional negative association with subsequent factor returns, whereas the return-correlation proxy does not. The two proxies have limited linear comovement, but the evidence does not by itself identify their underlying economic objects.

当前状态仍是 **major revision before submission**，原因是稳健性边界、版本锚定、多重检验和排版终检尚未全部完成。

## 6. 新增的公开官方证据

已通过公开网页获取并保存到 `input/evidence/`：

- 上交所融资融券会员业务指南及 PIT 关键摘录；
- 深交所投资者问答页面及 PIT 关键摘录；
- FRL 官方投稿指南原始响应（当前访问返回保护页，具体字数仍需用户在浏览器中手动确认）。

两所交易所公开规则均支持“前一交易日数据在下一交易日开市前公布”的保守可得日设定。
因此当前代码中的下一交易日 `available_date` 规则有官方制度依据。

## 7. FRL 官方投稿指南核对结果（云电脑浏览器）

已在云电脑 Chrome 中核对 FRL 官方 Guide for Authors：

- 论文应少于 2,500 词；
- 摘要不得超过 250 词，并应说明目的、基本方法、主要发现及结论；
- 需要 1--7 个英文关键词；
- 投稿时必须提供 Highlights 独立可编辑文件；Highlights 为 3--5 个项目符号，每项最多 85 个字符（含空格）；
- 表格应作为可编辑文本提交，不应作为图片；
- 图像按文章出现顺序编号，建议作为独立文件提交；
- 需要提交 research data availability/data statement；若数据不能共享，需要说明原因；
- 需要披露 funding sources；
- 需要披露 competing interests；无利益冲突时选择 “I have nothing to declare.”；
- 需要在投稿时披露生成式 AI 在稿件准备过程中的使用；若使用，应在参考文献前添加声明。指南给出的模板要求说明工具/服务和使用原因，并声明作者审阅、编辑并对全文负责；
- 生成式 AI 不得列为作者或共同作者；
- 当前页面显示投稿费为 USD 200，且为不可退还费用；正式投稿前应再次确认金额和适用条件。

当前论文摘要约 150 词，符合 250 词上限；仍需补交独立 Highlights 文件，并把 AI disclosure、Data statement 和 Funding/COI 声明整理成投稿材料。
