# 项目归档说明

## 归档类型

本归档包含论文、代码、配置、派生数据、分析报告、投稿材料、官方证据和复现说明。

## 目录

- `paper/`：LaTeX 源文件、最终 PDF、表格、图、投稿信、Highlights、投稿声明和 checklist。
- `code/`：数据抓取、清洗、因子构造、拥挤度、回归、稳健性、功效、排版和流水线脚本。
- `data/clean/`：最终清洗面板、月频面板、因子收益、拥挤度、日频因子收益和元数据。
- `reports/`：审计报告、主表 CSV、稳健性结果、功效分析、闸门判定和数据字典。
- `input/evidence/`：两融 PIT 官方规则、FRL 投稿指南证据响应和证据清单。
- `input/`：研究规划、提案和投稿准备材料。
- `logs/`：流水线与各阶段运行日志。
- `TASKPLAN-2026-10-16.md`：项目执行排期和研究闸门记录。

## 重要复现命令

从项目根目录运行：

```bash
cd /home/user/金融科研1014
python3 code/09_regressions.py
python3 code/10_tail_risk.py
python3 code/11_robustness.py
python3 code/15_volatility.py
python3 code/13_power.py
python3 code/12_make_tables.py
```

完整流水线入口：

```bash
cd /home/user/金融科研1014/code
bash run_all.sh
```

## 当前核心结果

- comomentum：主证据单元 0/6 显著。
- margin-financing-balance proxy：主证据单元 6/6 负向显著。
- 结论是条件性预测关联和测量比较，不是因果效应或账户级持仓识别。

## 原始数据说明

`data/raw/` 约 1.3 GB，包含 Baostock 日线和交易所两融原始文件；`data/clean/` 约 460 MB，包含清洗面板和派生结果。原始数据的再分发需遵守上游数据提供方的条款。若只需查看论文和代码，可使用不含 `data/raw/` 的轻量归档。

## 归档校验

同目录中的 `checksums.sha256` 记录归档内文件的 SHA-256；下载后可在 Linux/macOS 中运行：

```bash
sha256sum -c checksums.sha256
```

## 投稿材料

优先查看：

1. `paper/main.pdf`
2. `paper/highlights.txt`
3. `paper/cover_letter.md`
4. `paper/frl_submission_statements.md`
5. `paper/submission-checklist.md`
6. `reports/revision_change_log_2026-10-16.md`


## GitHub repositories

- Research package (paper, code, cleaned/derived data, reports, evidence): https://github.com/WEN365-1/a-share-factor-crowding-research
- Raw data release (daily market and margin-financing files): https://github.com/WEN365-1/a-share-factor-crowding-raw-data/releases/tag/v1.0-data
