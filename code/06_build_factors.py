"""
06_build_factors.py —— 6 个风格因子的月度多空组合

因子定义（全部来自干净的日频 PIT 面板，不需要任何额外数据源）：
  VAL     价值     1/pbMRQ            （高 B/M 做多）
  SMB     规模     -ln(流通市值)       （小市值做多）
  MOM     动量     t-12 至 t-1 累计收益 （跳过最近 1 个月，A 股标准做法）
  QMJ     质量     隐含 ROE = pbMRQ/peTTM （高 ROE 做多）
  LOWVOL  低波     -过去 60 交易日收益标准差（低波做多）
  TURN    换手     -过去 20 交易日平均换手率（低换手做多）

构造：每月最后可见交易快照调仓，按当时 isST 状态和因子值分十分位，流通市值加权。
      多空收益 = 顶部分位的加权收益 － 底部分位的加权收益。
      权重用 t 月末的流通市值（避免用未来权重）；历史与持有期 ST 交易日不删除。

产出：
  data/clean/monthly.parquet        月频截面：每只股票每月的 6 个因子暴露 + 当月/次月收益 + 流通市值
  data/clean/factor_returns.parquet 因子收益序列：date × factor 的多空/多头/空头收益、成分股数
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, CFG, Gate, REPORTS, formation_universe, get_logger, heartbeat,  # noqa: E402
                    now_local, realize_leg_return, write_parquet)

log = get_logger("06_factors", "06_build_factors.log")


def load_panel() -> pd.DataFrame:
    files = sorted((CLEAN / "panel").rglob("*.parquet"))
    log.info(f"读取面板分片 {len(files)} 个")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["code"] = df["code"].astype(str)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    log.info(f"面板：{len(df):,} 行，{df['code'].nunique()} 只股票，{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def main() -> None:
    gate = Gate(log)
    cfg = CFG
    n_groups = cfg["factors"]["n_groups"]
    log.info(f"开始构造因子 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    p = load_panel()

    # ---------- 1. 日频滚动统计（只做两个窗口）----------
    log.info("计算 60 日波动率 与 20 日平均换手率")
    g = p.groupby("code", observed=True)
    p["vol60"] = g["ret"].rolling(60, min_periods=60).std().reset_index(level=0, drop=True)
    p["turn20"] = g["turn"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    del g

    # ---------- 2. 月度聚合 ----------
    log.info("聚合到月度")
    p["ym"] = p["date"].dt.to_period("M")
    p["gross"] = 1.0 + p["ret"]
    # 月度收益必须要求该月所有保留交易日都有有效收益；未知收益不再静默当作零收益。
    p["ret_missing"] = p["ret"].isna()
    mret = (p.groupby(["code", "ym"], observed=True)
              .agg(gross=("gross", lambda s: s.prod(min_count=len(s))),
                   n_ret=("ret", "count"),
                   n_days_obs=("date", "count")))
    mret["mret"] = mret["gross"] - 1.0
    mret.loc[mret["n_ret"] < mret["n_days_obs"], "mret"] = np.nan
    mret = mret["mret"].rename("mret")

    # 每月最后一个交易日那一行作为截面暴露；同时记录个股实际交易天数与
    # 当月交易日历天数，用于施加「非交易日超过 10 天」的统一剔除规则。
    n_days = p.groupby(["code", "ym"], observed=True)["date"].count().rename("n_days")
    cal = pd.read_parquet(CLEAN / "calendar.parquet", columns=["date"])
    cal["date"] = pd.to_datetime(cal["date"])
    cal["ym"] = cal["date"].dt.to_period("M")
    calendar_days = cal.groupby("ym", observed=True)["date"].nunique().rename("calendar_days")
    last_idx = p.groupby(["code", "ym"], observed=True)["date"].idxmax()
    snap = p.loc[last_idx, ["code", "ym", "date", "close", "float_mv", "turn", "vol60", "turn20",
                            "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "industry", "isST"]].copy()
    snap = snap.merge(n_days.reset_index(), on=["code", "ym"], how="left")
    snap = snap.merge(calendar_days.reset_index(), on=["ym"], how="left")
    snap = snap.merge(mret.reset_index(), on=["code", "ym"], how="left")

    # ---------- 3. 因子暴露 ----------
    pb = snap["pbMRQ"].replace(0, np.nan)
    pe = snap["peTTM"].replace(0, np.nan)
    snap["VAL"] = np.where(pb > 0, 1.0 / pb, np.nan)              # B/M
    snap["SMB"] = -np.log(snap["float_mv"].clip(lower=1.0))      # 小市值做多
    snap["QMJ"] = np.where((pb > 0) & (pe > 0), pb / pe, np.nan)  # 隐含 ROE = E/B
    snap["LOWVOL"] = -snap["vol60"]
    snap["TURN"] = -snap["turn20"]

    # MOM：t-12 至 t-1 的累计收益（月度收益复利，跳过最近 1 个月）
    #
    # 必须先把每只股票补齐到完整的月度网格再滚动。原来的写法是在"已存在的行"上滚动，
    # 而**整月停牌的月份在面板里根本没有行**（05 只保留 tradestatus==1 的日），
    # 于是 rolling(12) 数的是 12 个观测而不是 12 个月——停牌股的时间跨度会被拉长，
    # 动量窗口与其他股票不可比。补齐后缺失月为 NaN，min_periods=12 自然要求
    # 连续 12 个自然月都有效。
    #
    # 另外 min_periods 必须等于 12（原来是 9）。允许 9 个月的部分窗口会让
    # 上市不足一年的股票带着"方差更小、跨度更短"的动量值参与排序，
    # 而这类股票恰恰最容易被推进极端分位。
    snap = snap.sort_values(["code", "ym"])
    m = snap.set_index(["code", "ym"])["mret"]
    full_idx = pd.MultiIndex.from_product(
        [snap["code"].unique(),
         pd.period_range(snap["ym"].min(), snap["ym"].max(), freq="M")],
        names=["code", "ym"])
    m = np.log1p(m.reindex(full_idx).clip(-0.99, None))
    lr12 = (m.groupby(level="code").shift(1)
            .groupby(level="code").rolling(12, min_periods=12).sum()
            .reset_index(level=0, drop=True))
    snap = snap.merge(np.expm1(lr12).rename("MOM").reset_index(), on=["code", "ym"], how="left")

    # ---------- 4. 次月收益（被解释变量）----------
    snap["ym_next"] = snap["ym"] + 1
    nxt = snap[["code", "ym", "mret"]].rename(columns={"ym": "ym_next", "mret": "ret_next"})
    snap = snap.merge(nxt, on=["code", "ym_next"], how="left")

    FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
    keep = ["date", "ym", "code", "industry", "float_mv", "n_days", "calendar_days",
            "isST", "mret", "ret_next"] + FACTORS
    monthly = snap[keep].copy()
    write_parquet(monthly, CLEAN / "monthly.parquet")
    log.info(f"月频截面：{len(monthly):,} 行 × {monthly['ym'].nunique()} 个月")

    # ---------- 5. 分组构造多空组合 ----------
    log.info(f"分 {n_groups} 组构造多空组合（流通市值加权）")
    recs = []
    leg_rows = []
    for ym, gdf in monthly.groupby("ym", observed=True):
        if not gate.ok():
            log.warning("闸门触发，因子构造提前收手")
            break
        # 形成月只使用当时可见的市值、交易状态和因子暴露；ret_next 不参与筛选。
        base = formation_universe(gdf)
        if len(base) < n_groups * 5:
            continue
        for f in FACTORS:
            d = base.dropna(subset=[f])
            if len(d) < n_groups * 5:
                continue
            try:
                d = d.assign(grp=pd.qcut(d[f].rank(method="first"), n_groups, labels=False))
            except ValueError:
                continue
            top = d[d["grp"] == n_groups - 1]
            bot = d[d["grp"] == 0]
            w_t = top["float_mv"] / top["float_mv"].sum()
            w_b = bot["float_mv"] / bot["float_mv"].sum()
            # 成员和原始权重在 t 月冻结。实现收益时仅在收益非缺失者上按腿
            # 重新归一化，并披露每腿被保留的形成月权重（收益覆盖率）。
            leg_rows.append(top[["code"]].assign(ym=ym, factor=f, side="top",
                                                 w=w_t.to_numpy()))
            leg_rows.append(bot[["code"]].assign(ym=ym, factor=f, side="bot",
                                                 w=w_b.to_numpy()))

            r_t, cov_t, n_ret_t = realize_leg_return(top, w_t)
            r_b, cov_b, n_ret_b = realize_leg_return(bot, w_b)
            recs.append({"ym": str(ym), "date": d["date"].max(), "factor": f,
                         # 口径提醒：本字段在 ym=t 记录、但实现于 t+1 月（用的是 ret_next）。
                         # 因此 ls_fwd1 本身就是「未来 1 个月收益」，下游做预测回归时
                         # **不要再 shift(-1)**；未来 h 个月的累计收益 = 连续 h 个 ls_fwd1 之和。
                         "top_ret": r_t, "bot_ret": r_b, "ls_fwd1": r_t - r_b,
                         "n_all": int(len(d)), "n_long": int(len(top)), "n_short": int(len(bot)),
                         "n_long_ret": n_ret_t, "n_short_ret": n_ret_b,
                         "top_ret_coverage": cov_t, "bot_ret_coverage": cov_b,
                         "top_mv_share": float(top["float_mv"].sum() / d["float_mv"].sum())})

    fr = pd.DataFrame(recs)
    fr["date"] = pd.to_datetime(fr["date"])
    write_parquet(fr, CLEAN / "factor_returns.parquet")

    # ---------- 6. 日频多空收益 ----------
    # 为什么要单独算这一份：Lou & Polk 在美股还记录了 comomentum 预测因子的
    # **实现波动与偏度**。日频多空收益因此用于不同经济原假设下的补充诊断；
    # 二阶矩结果无论显著与否，都不能单独验证或证伪 comomentum 的实现。
    log.info("计算日频多空收益（供波动/偏度检验用）")
    # 注意用 concat 而不是 pd.DataFrame(list)：后者会把 DataFrame 列表当成 3 维数组
    leg = pd.concat(leg_rows, ignore_index=True)
    leg["hold_ym"] = leg["ym"] + 1
    dd = p[["code", "date", "ym", "ret"]].rename(columns={"ym": "hold_ym"})
    dd["code"] = dd["code"].astype(str)
    leg["code"] = leg["code"].astype(str)
    mg = leg.merge(dd, on=["hold_ym", "code"], how="inner")
    mg["wr"] = (mg["w"].astype("float32") * mg["ret"].astype("float32")).astype("float32")
    byside = (mg.groupby(["ym", "factor", "side", "date"], observed=True)["wr"]
              .sum().unstack("side"))
    byside = byside.dropna()
    if "top" in byside.columns and "bot" in byside.columns:
        daily = pd.DataFrame({
            "ym": byside.index.get_level_values("ym"),
            "factor": byside.index.get_level_values("factor"),
            "date": byside.index.get_level_values("date"),
            "ls_fwd1": (byside["top"] - byside["bot"]).to_numpy(),
        }).sort_values(["factor", "date"])
        write_parquet(daily, CLEAN / "factor_daily.parquet")
        log.info(f"日频多空收益：{len(daily):,} 行（{daily['factor'].nunique()} 个因子 × "
                 f"{daily['date'].nunique()} 个交易日）")
    else:
        log.warning("日频多空收益构造失败：长/短两端未同时出现")
    log.info(f"因子收益表：{len(fr)} 行（{fr['ym'].nunique()} 个月 × {fr['factor'].nunique()} 个因子）")

    # ---------- 6. 速览 ----------
    summ = fr.groupby("factor")["ls_fwd1"].agg(["mean", "std", "count"])
    summ["年化收益"] = summ["mean"] * 12
    summ["年化波动"] = summ["std"] * np.sqrt(12)
    summ["夏普"] = summ["mean"] / summ["std"] * np.sqrt(12)
    log.info("\n" + summ.round(4).to_string())
    summ.to_csv(REPORTS / "factor_returns_summary.csv", encoding="utf-8-sig")
    heartbeat("06 因子构造结束", log)


if __name__ == "__main__":
    main()
