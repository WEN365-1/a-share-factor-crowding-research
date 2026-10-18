"""
05_build_panel.py —— 装配干净面板

做四件事：
  1. 把 5418 个个股 parquet 合并成按年分区的面板
  2. 推导流通股本与流通市值：换手率(%) = 成交量 / 流通股本 × 100
  3. 应用 01-Research-Proposal 3.2 节的剔除规则
  4. 落一份数据字典

**内存设计**：不把全部数据留在内存里。分两趟：
  第一趟：读一批（BATCH 个文件）→ 清洗 → 推导 → 写成一个临时 parquet（带 year 列）
  第二趟：按年份用 pyarrow 谓词下推只读该年的行，施加剔除规则，写出分区
  峰值内存 = 一批的量，而不是全样本的量。8GB 机器上这是必须的。

产出：
  data/clean/panel/year=YYYY/part.parquet
  reports/data_dictionary.md
"""
from __future__ import annotations

import sys
from pathlib import Path
import gc
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, CFG, DAILY, DATA, Gate, REPORTS, get_logger,  # noqa: E402
                    heartbeat, read_parquet, write_parquet, now_local)

log = get_logger("05_panel", "05_build_panel.log")

NUM_COLS = ["open", "high", "low", "close", "preclose", "volume", "amount",
            "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
CARRY = ["date", "code", "close", "ret", "volume", "amount", "turn",
         "float_mv", "shares_float", "industry", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
         "tradestatus", "isST", "year"]
BATCH = 400

# 金融行业的识别关键词。用**包含**匹配而不是精确匹配——Baostock 的行业名带证监会
# 门类前缀（「J66货币金融服务」「J68保险业」…），精确匹配不会命中任何一只。
# 这些词同时覆盖带前缀与不带前缀两种写法，换数据源也不会失效。
FINANCIAL_KEYS_RE = ("货币金融|资本市场|保险|其他金融|租赁|多元金融|银行|证券|信托|基金")


def prep_one(f, ipo_map, ind_map, cfg, log) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(f)
    except Exception as e:  # noqa: BLE001
        log.warning(f"读取失败 {f.name}: {e}")
        return None
    if d.empty:
        return None
    d = d.drop(columns=[c for c in ("adjustflag",) if c in d.columns])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in NUM_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tradestatus"] = pd.to_numeric(d["tradestatus"], errors="coerce").fillna(0).astype("int8")
    d["isST"] = pd.to_numeric(d["isST"], errors="coerce").fillna(0).astype("int8")
    # 刻意不用 category dtype：各文件分类基数不同，concat 时会退化成 object，行为难预测
    d["code"] = d["code"].astype(str)

    # ---- 推导流通股本与流通市值 ----
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(d["turn"].to_numpy() > 0,
                          d["volume"].to_numpy() * 100.0 / d["turn"].to_numpy(), np.nan)
    s = pd.Series(shares, index=d.index).replace([np.inf, -np.inf], np.nan).ffill()
    d["shares_float"] = s
    d["float_mv"] = d["shares_float"] * d["close"]
    d["ret"] = d["pctChg"] / 100.0
    d["industry"] = d["code"].map(ind_map)
    d["year"] = d["date"].dt.year.astype("int16")

    keep = [c for c in CARRY if c in d.columns]
    out = d[keep].copy()
    for c in ["close", "ret", "volume", "amount", "turn", "float_mv",
              "shares_float", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        if c in out.columns:
            out[c] = out[c].astype("float32")
    del d
    return out


def main() -> None:
    cfg = CFG
    gate = Gate(log)
    log.info(f"开始装配面板（分批，批大小 {BATCH}）| 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    uni = read_parquet(CLEAN / "universe.parquet")
    ind = read_parquet(CLEAN / "industry.parquet")
    ipo_map = uni.set_index("code")["ipoDate"].to_dict()
    ind_map = (ind[["code", "industry"]].drop_duplicates("code").set_index("code")["industry"].to_dict()
               if len(ind) else {})

    files = sorted((DAILY / "3").rglob("*.parquet"))
    log.info(f"待合并文件 {len(files)} 个")
    if not files:
        log.error("没有日线文件，退出")
        return

    tmp = DATA / "tmp_panel"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    # ---------- 第一趟：分批清洗并落临时文件 ----------
    buf, nb, n_ok = [], 0, 0
    for i, f in enumerate(files):
        if not gate.ok():
            log.warning("闸门触发，面板装配第一趟提前收手")
            break
        out = prep_one(f, ipo_map, ind_map, cfg, log)
        if out is not None and len(out):
            buf.append(out)
            n_ok += 1
        if len(buf) >= BATCH:
            pd.concat(buf, ignore_index=True).to_parquet(tmp / f"batch{nb:04d}.parquet", index=False)
            nb += 1
            buf = []
            gc.collect()
        if (i + 1) % 500 == 0:
            log.info(f"第一趟 {i + 1}/{len(files)}（有效 {n_ok}），已落 {nb} 批")
    if buf:
        pd.concat(buf, ignore_index=True).to_parquet(tmp / f"batch{nb:04d}.parquet", index=False)
        nb += 1
    log.info(f"第一趟完成：有效文件 {n_ok}，临时批 {nb} 个")

    # ---------- 第二趟：按年施加剔除规则并写出 ----------
    batches = sorted(tmp.glob("batch*.parquet"))
    years: set[int] = set()
    for b in batches:
        try:
            col = pq.read_table(b, columns=["year"]).column("year").to_numpy()
            years.update(int(x) for x in np.unique(col))
        except Exception:  # noqa: BLE001
            continue
    yrs = sorted(years)
    log.info(f"覆盖年份 {yrs}")

    total, summary = 0, []
    for yr in yrs:
        if not gate.ok():
            log.warning("闸门触发，第二趟提前收手")
            break
        parts = []
        for b in batches:
            try:
                t = pq.read_table(b, filters=[("year", "==", yr)])
            except Exception:  # noqa: BLE001
                continue
            if t.num_rows:
                parts.append(t.to_pandas())
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()
        n0 = len(df)

        df = df[df["tradestatus"] == 1]                     # 只保留正常交易
        ipo_dt = pd.to_datetime(df["code"].map(ipo_map), errors="coerce")
        df = df[(df["date"] - ipo_dt).dt.days >= cfg["universe"]["min_list_days"]]
        if cfg["universe"]["drop_st"]:
            df = df[df["isST"] == 0]
        if cfg["universe"]["drop_financial"]:
            # 注意：Baostock 的行业名带证监会门类前缀，实际值是
            # 「J66货币金融服务」「J67资本市场服务」「J68保险业」「L71租赁业」这种形式。
            # 用精确匹配 {"银行","保险",...} **永远不会命中**——实测确实一只金融股都没剔掉。
            # 必须用关键词包含匹配。金融股是估值、杠杆与两融余额都最特殊的一类，
            # 漏掉它们会直接扭曲 VAL / QMJ / SMB。
            ind_str = df["industry"].fillna("").astype(str)
            is_fin = ind_str.str.contains(FINANCIAL_KEYS_RE, regex=True)
            if is_fin.any():
                log.info(f"  剔除金融行业 {int(is_fin.sum()):,} 行"
                         f"（行业名示例：{sorted(ind_str[is_fin].unique())[:4]}）")
            df = df[~is_fin]

        cols = [c for c in ["date", "code", "close", "ret", "volume", "amount", "turn",
                            "float_mv", "shares_float", "industry", "peTTM", "pbMRQ",
                            "psTTM", "pcfNcfTTM"] if c in df.columns]
        out = df[cols].sort_values(["date", "code"])
        d_dir = CLEAN / "panel" / f"year={yr}"
        d_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(out, d_dir / "part.parquet")
        pct = (n0 - len(out)) / max(n0, 1)
        log.info(f"年份 {yr}：{n0:,} → {len(out):,} 行（剔除 {n0 - len(out):,}，{pct:.1%}）")
        summary.append({"year": yr, "before": n0, "after": len(out)})
        total += len(out)
        del df, out
        gc.collect()

    shutil.rmtree(tmp, ignore_errors=True)

    pd.DataFrame(summary).to_csv(REPORTS / "panel_build_summary.csv", index=False, encoding="utf-8-sig")
    meta = {"rows": int(total), "years": yrs, "built_at": now_local().strftime("%Y-%m-%d %H:%M:%S")}
    (CLEAN / "panel_meta.json").write_text(pd.Series(meta).to_json(force_ascii=False, indent=2),
                                           encoding="utf-8")

    dd = """# 数据字典 · 干净面板

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
"""
    (REPORTS / "data_dictionary.md").write_text(dd, encoding="utf-8")
    heartbeat(f"05 面板装配结束（{total:,} 行）", log)


if __name__ == "__main__":
    main()
