# 数据字典 · 干净面板

| 字段 | 含义 | 来源与口径 |
|---|---|---|
| date | 交易日 | Baostock |
| code | 股票代码（sh.600000 / sz.000001） | Baostock |
| close | 收盘价（**不复权**） | Baostock adjustflag=3 |
| ret | 日收益率 = pctChg/100 | Baostock pctChg（以交易所除权除息后昨收为基准，**已含分红送转**） |
| volume | 成交量（股） | Baostock |
| amount | 成交额（元） | Baostock |
| turn | 换手率（%） | Baostock |
| shares_float | 流通股本（股），由 volume×100/turn 推导，停牌日以前值填充 | 推导值 |
| float_mv | 流通市值 = shares_float × close | 推导值 |
| industry | 行业 | Baostock query_stock_industry |
| peTTM | 市盈率 TTM | Baostock，**PIT（按公告日更新，已实证验证）** |
| pbMRQ | 市净率 MRQ | Baostock，**PIT** |
| psTTM | 市销率 TTM | Baostock，**PIT** |
| pcfNcfTTM | 市现率 TTM | Baostock，**PIT** |

## 剔除规则（已应用）
1. tradestatus ≠ 1（停牌等）
2. 上市不足 180 天
3. ST / *ST
4. 金融行业（银行、保险、证券、多元金融、资本市场、租赁）
5. 北交所（Baostock 不覆盖）

## 未在本层应用、留到因子层的规则
- 当月停牌超过 10 个交易日 → 在 06 用 n_days 施加
- 月内日收益缺失超过 20% → 由 tradestatus 过滤后自然满足
- 从未进入两融标的名单 → **刻意不剔除**：因子组合在全 A 构造，
  两融拥挤度只在成分股上求和，并另设「两融水平」「组合两融覆盖率」控制变量（见 08/09）

## 前视偏差说明
价 / 量 / 换手 / 估值字段均取自当日可得的公开行情，不含未来信息。
收益率使用不含未来信息的 pctChg。估值字段的 PIT 性已由实证检验确认：
浦发银行的每股净资产（由 close/pbMRQ 反推）只在 2024-04-30 与 2024-08-20 换台阶，
即年报/一季报与半年报的**披露日**，而非报告期末。

## 2026-10-18 增补（投稿前补强轮）

### 新增派生字段
| 字段 | 定义 | 说明 |
|---|---|---|
| crowd_margin_short | crowd_margin_long − crowd_margin | 空头腿两融强度聚合（v_short），由 L−S 恒等式反推，见 11/16 |

### 两融覆盖率（各因子差异，供正文披露）
| factor | cover_long 均值 | 最低 | <0.30 月份占比 | cover_short 均值 |
|---|---|---|---|---|
| VAL | 0.922 | 0.763 | 0% | 0.753 |
| SMB | 0.255 | 0.000 | 61.1% | 0.980 |
| MOM | 0.821 | 0.331 | 0% | 0.758 |
| QMJ | 0.903 | 0.745 | 0% | 0.739 |
| LOWVOL | 0.922 | 0.788 | 0% | 0.594 |
| TURN | 0.933 | 0.835 | 0% | 0.503 |

（来源：reports/table_cov_stats.csv，由 16_extension_analyses.py 生成。
config 的 coverage_floor: 0.30 未在 08 强制执行，现作为稳健性检验在 16 中施加。）

### PIT 审计升级为多股票
原单股审计（浦发银行）已推广为 40 只股票审计：reports/pit_audit_multistock.md +
pit_audit_multistock.csv（脚本 17_pit_audit.py）。正文「multi-stock audit」说法以此为准。

### 本地可复现性
全部分析脚本可在 Windows 本地重跑（FIN_ROOT 环境变量覆盖根目录；
硬编码的 /home/user 路径已全部改为 Path(__file__) 可移植写法）。
2026-10-18 实测 09/10/11/15 重跑结果与云主机基线逐位一致（最大差异 ≤1e-14）。

### 流量型代理（2026-10-18 增补）
| 字段 | 定义 | 说明 |
|---|---|---|
| crowd_flow_buy | 月度融资买入强度（∑fin_buy / 流通市值）的腿级流通市值加权多空差 | 两所都有 fin_buy；期间量按月求和（按 T+1 可用日归月） |
| crowd_flow_net | 同上，但用 ∑fin_buy − ∑fin_repay | 仅 SSE 公布 fin_repay；聚合权重在 SSE 成分股上重新归一化 |
| crowd_flow_{buy,net}_level | 上述测度的腿级 12 个月滚动均值（滞后 1 个月） | 流量自身的水平控制 |

产出：reports/table11_flow_proxy.csv、table11b_horserace.csv、flow_proxy_corr.csv（脚本 18_flow_proxy.py）。
