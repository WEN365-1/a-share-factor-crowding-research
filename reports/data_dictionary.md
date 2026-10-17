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
