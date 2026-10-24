# 因子拥挤研究归档与复现说明

## 当前版本

2026-10-25 路径 B 公开版。论文定位是**收益相关代理与可观测融资代理的测度比较**，不是账户级
总持仓识别、因果估计或美国结论的单国复制。`paper/main.tex` 是当前稿件；`paper/main.pdf`
是编译检查版。当前复现边界和公开闸门以根目录 `README.md` 与投稿检查清单为准。

代码采用 MIT 许可；文稿及其他原创文档暂时保留所有权利。MIT 不覆盖文稿、第三方数据或
供应方派生数据集。

## 目录

- `paper/`：LaTeX、表格、图、投稿信、highlights、声明和检查清单。
- `code/`：数据抓取、清洗、因子与测度构造、回归、稳健性、功效、测试和流水线。
- `reports/data_dictionary.md`：输入字段和构造口径文档，不是数据文件。
- `TASKPLAN-2026-10-16.md`：早期研究计划；其中旧结论和闸门文字不等于当前论文结论。

路径 B 不分发独立的原始、证券级、组合级或其他供应方派生数据集，也不分发独立结果
CSV／JSON。论文结果表和图作为文稿组成部分保留，不作为可下载数据集提供。

## 路径与环境

所有分析脚本通过环境变量 `FIN_ROOT` 定位项目根目录。不要依赖早期硬编码的
`/home/user/金融科研1014`。根目录 `README.md` 给出 Windows 与 Linux/macOS 的安装和
测试命令。`requirements-analysis.txt` 锁定分析依赖，`requirements.txt` 增加抓取依赖。

2026-10-24 的本地路径 A 验证曾使用四个组合级输入，在仅含 23 个锁定依赖的新 Python 3.12
环境中通过 45+4 项测试，并完成 10 个脚本及 29 个输出的逐字节核验。这是本地验证历史；
当前路径 B 不包含这些输入或独立结果文件，公开包本身不能独立复现实证结果。缺少相应
输入的集成测试必须明确跳过，不能把跳过写成通过。

## 完整流水线

```bash
FIN_ROOT=/absolute/path/to/a-share-factor-crowding-research bash code/run_all.sh
```

流水线包含 01--30 号当前分析阶段，需要研究者依供应方条款自行取得并构建相应输入。
`12_make_tables.py` 必须在相关实证脚本之后运行。编排器依赖 Bash 与 `flock`，不是
Windows 一键脚本；只有实际运行完整流水线时才生成 `data/run_manifest.json`。

## 新增核心检验

```bash
python3 code/19_comomentum_faithful.py
python3 code/20_comomentum_faithful_tests.py
python3 code/21_comomentum_robustness.py
python3 code/22_comomentum_factor_slopes.py
python3 code/23_lp_second_moments.py
python3 code/24_margin_staleness_robustness.py
python3 code/25_margin_staleness_tests.py
```

`24_margin_staleness_robustness.py` 先要求 45 天重建的五项融资字段与既有
`crowding.parquet` 逐项一致，再运行 45/90 天的同口径规格 D 回归；两个阈值共享形成月
股票池、T+1 规则、缺失值处理、控制变量和循环平移估计量。

## 当前证据边界

- 融资余额变化与 pooled 后续因子收益呈条件性负向关联，主要体现在低侧腿；因子和子样本
  结果并不普遍显著，不能写成“所有因子稳定衰减”。
- 基线日频 comomentum 的短持有期关联有限；周频 LP 对照出现探索性的长期结果，但对构造、
  因子和多重检验敏感，不能写成“完全无预测力”或“完整复制 Lou--Polk”。
- 偏度、波动和坏周证据不一致，既不证明实现失败，也不验证任一代理就是实际拥挤持仓。
- 所有经济结果均为关联性证据。六因子、约 120 个月、当前行业快照和融资代理的制度含义
  仍是主要限制。
- 早期 `reports/gate_*`、修订日志和任务计划可能保留旧表述；只有当前 `main.tex` 与本轮
  重跑输出可以用于投稿结论。6/24 个月窗口已复用主分析的 T+1、45 天阈值、权重、覆盖率
  和缺失规则；h=12 的观测与置换统一使用 HAC(12)，其他期限使用 HAC(6)。

## 原始数据与公开仓库

原始日线、交易所两融文件、证券级面板及组合级派生序列均不因“经过计算”而自动获得
再分发授权。路径 B 下，研究者须从上游供应方取得所需输入、遵守适用条款并自行构建
分析数据。研究仓库为：
https://github.com/WEN365-1/a-share-factor-crowding-research

2026-10-25 已完成路径 B 公开同步：raw-data release 在完整私有归档校验后移除，研究
仓库公开 HEAD 与完整文件树已读取核对，全新克隆通过 78 文件逐字节核验、23 依赖环境
检查、41 项适用测试（另 1 个数据依赖测试类明确跳过）、4 项融资专项测试和 10 页论文
完整编译。该范围不是独立实证数据复现；历史 raw-data 下载指引不代表当前资产状态或授权。
