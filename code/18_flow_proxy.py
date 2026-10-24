"""
18_flow_proxy.py —— 拓展：融资流量型拥挤代理（fin_buy / fin_repay）

动机（投稿前补强轮的「拓展」项）：
  主代理是**余额变化**（ΔMB/MV，存量视角）；原始两融数据还公开**逐日融资买入额
  fin_buy 与偿还额 fin_repay**（流量视角，Bian et al. 2026 的账户级流量文献的公开版）。
  流量代理 = 月度融资买入强度（∑fin_buy / 流通市值），多空腿之差；
  SSE 另有 fin_repay，可再算净流量（∑fin_buy − ∑fin_repay）。
  SZSE 不公布 fin_repay → 净流量仅 SSE 覆盖，作为辅助口径。

口径与 08/11 完全一致：
  - T+1 可用性（calendar 下一交易日）
  - 月度取数：月度内最后一个可用交易日的日度流量**求和**（流量是期间量，不是时点量）
  - 十分位归属：与 11 相同的 monthly.parquet 形成月非交易日规则、qcut、因子列
  - 腿聚合：流通市值加权
  - 推断：spec D 控制 + 因子固定效应；h=12 用 NW(12)，其余用 NW(6)；
    循环平移安慰剂使用相同滞后阶数（同 16.fit_row）

产出：reports/table11_flow_proxy.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, MARGIN, REPORTS, formation_universe, get_logger, hac_ols,  # noqa: E402
                    heartbeat, placebo_pvalue, read_parquet)

log = get_logger("18_flow", "18_flow_proxy.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
N_GROUPS = 10
HORIZONS = [1, 3, 6, 12]
MOM_CTRL = ["past1", "past6", "past12"]


def read_margin_flow_file(fp: Path) -> pd.DataFrame:
    """买入列必需；偿还列单独读取，允许深交所文件没有 fin_repay。"""
    schema_names = set(pq.read_schema(fp).names)
    required = {"date", "code", "fin_buy", "exchange"}
    missing = required.difference(schema_names)
    if missing:
        raise ValueError(f"{fp} 缺少必需流量字段: {sorted(missing)}")
    buy = pd.read_parquet(fp, columns=["date", "code", "fin_buy", "exchange"])
    if "fin_repay" not in schema_names:
        buy["fin_repay"] = np.nan
        return buy
    repay = pd.read_parquet(fp, columns=["date", "code", "fin_repay"])
    return buy.merge(repay, on=["date", "code"], how="left", validate="one_to_one")


def aggregate_flow_monthly(mg: pd.DataFrame) -> pd.DataFrame:
    """月度流量求和；全缺失偿还额保持 NaN，净流量不会被误算成买入额。"""
    g = mg.groupby(["code", "ym"], observed=True)
    fl = g.agg(fin_buy_sum=("fin_buy", lambda s: s.sum(min_count=1)),
               fin_repay_sum=("fin_repay", lambda s: s.sum(min_count=1)),
               n_days_flow=("date", "nunique")).reset_index()
    fl["fin_net_sum"] = np.where(fl["fin_repay_sum"].notna(),
                                 fl["fin_buy_sum"] - fl["fin_repay_sum"], np.nan)
    return fl


def fit_row(d: pd.DataFrame, cols: list[str], measure: str, y: str,
            meta: dict) -> dict | None:
    d = d.dropna(subset=cols + [y]).sort_values(["ym", "factor"]).copy()
    if len(d) < 40:
        return None
    lags = 12 if meta.get("horizon") == 12 else 6
    r = hac_ols(d[y].to_numpy(), d[cols].to_numpy(), cols, lags=lags)
    pp, _ = placebo_pvalue(d, cols, measure, y, r[measure]["t"], hac_lags=lags)
    row = {"measure": measure.replace("_z", ""), "hac_lags": lags, "beta": r[measure]["coef"],
           "se": r[measure]["se"], "t": r[measure]["t"], "p": r[measure]["p"],
           "p_placebo": pp, "n": r["_n"]}
    row.update(meta)
    return row


def build_flow_panel() -> pd.DataFrame:
    """构造流量型代理面板（买入流量；SSE 另算净流量），与 16 的 build_panel 口径一致。"""
    # ---------- 1. 原始两融日度流量（T+1 可用性） ----------
    files = sorted(MARGIN.rglob("*.parquet"))
    parts = []
    for fp in files:
        x = read_margin_flow_file(fp)
        if len(x):
            parts.append(x)
    if not parts:
        raise ValueError("没有可读取的融资买入流量数据")
    mg = pd.concat(parts, ignore_index=True)
    mg["date"] = pd.to_datetime(mg["date"])
    mg["code"] = mg["code"].astype(str)
    mg["exchange"] = mg["exchange"].astype(str)
    for c in ("fin_buy", "fin_repay"):
        mg[c] = pd.to_numeric(mg[c], errors="coerce")
    cal = pd.DatetimeIndex(pd.to_datetime(
        read_parquet(CLEAN / "calendar.parquet")["date"])).sort_values().unique()
    next_day = pd.Series(cal[1:], index=cal[:-1])
    mg["available_date"] = mg["date"].map(next_day)
    mg = mg.dropna(subset=["available_date"])
    mg["ym"] = mg["available_date"].dt.to_period("M")
    # 流量是期间量；偿还额必须 min_count=1，避免深交所全 NaN 被聚合为零。
    fl = aggregate_flow_monthly(mg)
    coverage = (mg.groupby(["exchange", "ym"], observed=True)
                  .agg(stock_months=("code", "size"), stocks=("code", "nunique"),
                       buy_observations=("fin_buy", "count"),
                       repay_observations=("fin_repay", "count")).reset_index())
    coverage["buy_coverage"] = coverage["buy_observations"] / coverage["stock_months"]
    coverage["repay_coverage"] = coverage["repay_observations"] / coverage["stock_months"]
    coverage.to_csv(REPORTS / "flow_exchange_coverage.csv", index=False, encoding="utf-8-sig")

    # ---------- 2. 十分位归属（与 06/08/11 相同规则） ----------
    monthly = read_parquet(CLEAN / "monthly.parquet")
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    monthly2 = monthly[["ym", "code", "float_mv", "n_days", "calendar_days", "isST"] + FACTORS].copy()
    monthly2 = formation_universe(monthly2)
    monthly2["ym"] = monthly2["ym"].astype(str)
    fl["ym"] = fl["ym"].astype(str)
    mm = monthly2.merge(fl, on=["code", "ym"], how="left")
    mem_rows = []
    for ym, g in monthly2.groupby("ym", observed=True):
        for f in FACTORS:
            d = g.dropna(subset=[f])
            if len(d) < N_GROUPS * 5:
                continue
            try:
                d = d.assign(grp=pd.qcut(d[f].rank(method="first"), N_GROUPS, labels=False))
            except ValueError:
                continue
            mem_rows.append(d[["ym", "code", "float_mv"]].assign(factor=f, grp=d["grp"].values))
    mem = pd.concat(mem_rows, ignore_index=True)
    mm["fin_buy_int"] = np.where(mm["float_mv"] > 0, mm["fin_buy_sum"] / mm["float_mv"], np.nan)
    mm["fin_net_int"] = np.where(mm["float_mv"] > 0, mm["fin_net_sum"] / mm["float_mv"], np.nan)
    mgm = mem.merge(mm[["code", "ym", "fin_buy_int", "fin_net_int"]],
                    on=["code", "ym"], how="left")

    # ---------- 3. 腿级聚合（流通市值加权强度，多空之差） ----------
    # 权重在**有数据**的成分股上重新归一化：SZSE 不公布 fin_repay，
    # 净流量只有 SSE 成分股参与聚合，若用全腿权重会系统性低估净流量强度。
    rows = []
    for (ym, f), g in mgm.groupby(["ym", "factor"], observed=True):
        top = g[g["grp"] == N_GROUPS - 1]
        bot = g[g["grp"] == 0]
        if len(top) < 10 or len(bot) < 10:
            continue
        def leg_int(grp, col):
            v = grp[col]
            m = v.notna()
            if m.sum() == 0:
                return np.nan
            w = grp.loc[m, "float_mv"]
            w = w / w.sum()
            return float((v[m] * w).sum())
        rows.append({"ym": ym, "factor": f,
                     "crowd_flow_buy": leg_int(top, "fin_buy_int") - leg_int(bot, "fin_buy_int"),
                     "crowd_flow_net": leg_int(top, "fin_net_int") - leg_int(bot, "fin_net_int")})
    cw = pd.DataFrame(rows)
    cw["ym"] = pd.PeriodIndex(cw["ym"], freq="M")
    # 流量自身的水平控制：腿级流量强度过去 12 个月均值（滞后 1 个月，与因子动量口径一致）
    cw = cw.sort_values(["factor", "ym"])
    for c in ("crowd_flow_buy", "crowd_flow_net"):
        cw[c + "_level"] = (cw.groupby("factor", observed=True)[c]
                            .shift(1).rolling(12, min_periods=12).mean())

    # ---------- 4. 与主面板合并、标准化、固定效应 ----------
    fr = read_parquet(CLEAN / "factor_returns.parquet")
    fr["date"] = pd.to_datetime(fr["date"])
    fr = fr.sort_values(["factor", "date"])
    for k in (1, 6, 12):
        fr[f"past{k}"] = (fr.groupby("factor", observed=True)["ls_fwd1"]
                          .shift(1).rolling(k, min_periods=k).sum())
    for h in HORIZONS:
        fr[f"fwd{h}"] = (fr.groupby("factor", observed=True)["ls_fwd1"]
                         .transform(lambda s: s.rolling(h, min_periods=h).sum().shift(-(h - 1))))
    cr = read_parquet(CLEAN / "crowding.parquet")
    cr["ym"] = pd.PeriodIndex(cr["ym"], freq="M")
    fr["ym"] = fr["date"].dt.to_period("M")
    df = fr.merge(cr, on=["ym", "factor"], how="inner").merge(cw, on=["ym", "factor"], how="left")
    for c in ["comomentum", "crowd_margin", "crowd_flow_buy", "crowd_flow_net",
              "crowd_flow_buy_level", "crowd_flow_net_level",
              "level_margin", "cover_long", "cover_short"]:
        if c in df.columns:
            df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
    fe = pd.get_dummies(df["factor"], prefix="FE", drop_first=True).astype(float)
    df = pd.concat([df, fe], axis=1)
    return df.sort_values(["ym", "factor"]).copy()


def main() -> None:
    df = build_flow_panel()
    log.info(f"流量代理面板：{len(df)} 行；crowd_flow_buy 非空 {df['crowd_flow_buy'].notna().sum()} 行，"
             f"crowd_flow_net 非空 {df['crowd_flow_net'].notna().sum()} 行")
    ctrl = ([x for x in df.columns if x.startswith("FE_")] + MOM_CTRL
            + ["level_margin_z", "cover_long_z", "cover_short_z"])
    rows = []

    # 1) 单测度主回归（水平控制用流量自身 12 个月均值；净流量另加 SSE 口径说明）
    for measure, lvl in [("crowd_flow_buy_z", "crowd_flow_buy_level_z"),
                         ("crowd_flow_net_z", "crowd_flow_net_level_z")]:
        if measure not in df.columns:
            continue
        for h in HORIZONS:
            r = fit_row(df, [measure] + ctrl + [lvl], measure, f"fwd{h}",
                        {"analysis": "flow proxy", "group": measure.replace("_z", ""),
                         "horizon": h})
            if r:
                rows.append(r)

    # 2) 赛马：流量与存量（余额变化）同时进入
    for measure, lvl in [("crowd_flow_buy_z", "crowd_flow_buy_level_z"),
                         ("crowd_flow_net_z", "crowd_flow_net_level_z")]:
        for h in HORIZONS:
            cols = [measure, "crowd_margin_z"] + ctrl + [lvl]
            r = fit_row(df, cols, measure, f"fwd{h}",
                        {"analysis": "horse race", "group": measure.replace("_z", ""),
                         "horizon": h})
            if r:
                rows.append(r)

    # 3) 相关性与有效性检查（对论文口径：两个流量测度与余额变化/ comomentum 的相关性）
    m = df[["comomentum", "crowd_margin", "crowd_flow_buy", "crowd_flow_net"]].corr()
    log.info("\n相关性矩阵：\n" + m.round(3).to_string())
    m.to_csv(REPORTS / "flow_proxy_corr.csv", index=True, encoding="utf-8-sig")
    sd = df[["crowd_flow_buy", "crowd_flow_net"]].std()
    log.info(f"流量测度 pooled SD：\n{sd.round(6).to_string()}")

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "table11_flow_proxy.csv", index=False, encoding="utf-8-sig")
    log.info("\n" + out.round(4).to_string())
    heartbeat("18 流量代理结束", log)


if __name__ == "__main__":
    main()
