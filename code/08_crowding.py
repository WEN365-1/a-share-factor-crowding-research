"""
08_crowding.py —— 两个因子拥挤度测度（本文核心）

测度 A（间接）：comomentum，严格按 Lou & Polk (2022) 的思路
  —— 因子多空两端成分股在形成期内「FF3 残差收益的两两平均相关」，
      用收益相关性间接推断有多少资金挤在同一个因子上。

测度 B（余额代理）：两融余额拥挤度
  —— 因子多空两端组合的融资余额变化强度，用公开可观测的杠杆融资余额近似资金拥挤。
      Crowd = Σ_{多头端} w_i · ΔMB_i / FloatMV_i  −  Σ_{空头端} w_i · ΔMB_i / FloatMV_i

并额外计算「组合两融覆盖率」——因为 2015 年可两融股不到全 A 的 1/3，
覆盖率本身随年份上升，若不加控制会把「样本变宽」误读成「拥挤度上升」。

产出：
  data/clean/crowding.parquet          月度：factor × ym × 两个测度 + 覆盖率
  reports/crowding_correlation.csv     两个测度的相关系数（Table 1 的核心发现）
  reports/fig_crowding_series.png      两个测度的时间序列（Figure 1 底稿）
"""
from __future__ import annotations

import gc
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/金融科研1014/code")
from common import CLEAN, CFG, Gate, MARGIN, REPORTS, get_logger, heartbeat, now_local, read_parquet, write_parquet  # noqa: E402

log = get_logger("08_crowding", "08_crowding.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
N_GROUPS = 10
SAMPLE_PER_LEG = 100      # 每端抽样股票数（全端两两相关计算量过大）
MIN_PER_LEG = 15          # 每端最少股票数：低于此两两相关矩阵噪声过大，当月不出测度
MIN_DAYS_LEG = 60         # 两端共同交易日最少天数
MIN_PAIRS = 60            # 两两相关所需的最少共同观测数
FORM_WINDOW = 250         # 形成期交易日数（约 12 个月）
MIN_WIN_DAYS = 120        # 窗口内最少交易日数，不足则当月不出测度
MIN_OBS_WINDOW = 120      # 单只股票在窗口内的最少有效交易日数，低于此不参与回归
SEED = 20261016


def load_daily() -> pd.DataFrame:
    """只读需要的四列。真实面板 1400 万行 × 14 列全读进来会吃掉 1.5GB 以上，
    而 08 只需要收益、流通市值和分组键。"""
    files = sorted((CLEAN / "panel").rglob("*.parquet"))
    d = pd.concat((pd.read_parquet(f, columns=["date", "code", "ret", "float_mv"])
                   for f in files), ignore_index=True)
    d["code"] = d["code"].astype(str)
    return d.sort_values(["date", "code"]).reset_index(drop=True)


def load_margin_monthly(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """按保守 T+1 可得规则构造月末两融信号。

    原始明细的 ``date`` 被视为发生日；研究者最早在下一个交易日看到，
    因此 ``available_date`` 是交易日历中的下一交易日。月末信号只使用
    ``available_date <= signal_date`` 的最新余额，避免把月末收盘后才披露
    的余额用于同一月末形成信号。余额变化比较相邻月末的可得快照，而不是
    两个不规则的原始观测日。
    """
    files = sorted(MARGIN.rglob("*.parquet"))
    log.info(f"读取两融明细 {len(files)} 个文件")
    parts = []
    for f in files:
        try:
            x = pd.read_parquet(f, columns=["date", "code", "margin_balance"])
        except Exception:  # noqa: BLE001
            continue
        if len(x):
            parts.append(x)
    if not parts:
        log.error("没有两融数据")
        return pd.DataFrame()

    mg = pd.concat(parts, ignore_index=True)
    mg["date"] = pd.to_datetime(mg["date"])
    mg["code"] = mg["code"].astype(str)
    mg["margin_balance"] = pd.to_numeric(mg["margin_balance"], errors="coerce")
    mg = mg.dropna(subset=["date", "code", "margin_balance"]).drop_duplicates(["code", "date"])

    cal = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values().unique()
    if len(cal) < 2:
        return pd.DataFrame()
    next_day = pd.Series(cal[1:], index=cal[:-1])
    mg["available_date"] = mg["date"].map(next_day)
    mg = mg.dropna(subset=["available_date"])

    month_ends = pd.Series(cal, index=cal).groupby(cal.to_period("M")).max()
    codes = mg["code"].unique()
    grid = pd.MultiIndex.from_product(
        [codes, month_ends.index], names=["code", "ym"]
    ).to_frame(index=False)
    grid["signal_date"] = grid["ym"].map(month_ends)

    right = mg[["code", "available_date", "date", "margin_balance"]].sort_values(
        ["available_date", "code"]
    )
    left = grid.sort_values(["signal_date", "code"])
    asof = pd.merge_asof(
        left, right, by="code", left_on="signal_date", right_on="available_date",
        direction="backward", allow_exact_matches=True,
    )
    asof = asof.sort_values(["code", "ym"])
    asof["d_mb"] = asof.groupby("code", observed=True)["margin_balance"].diff()
    asof["available_date"] = asof["available_date"].astype("datetime64[ns]")
    log.info(f"两融月末 PIT 面板：{len(asof):,} 行 | 覆盖 {asof['code'].nunique()} 只股票 | "
             f"{asof['ym'].nunique()} 个月；采用下一交易日可得规则")
    out = asof[["code", "ym", "signal_date", "available_date", "date",
                "margin_balance", "d_mb"]]
    audit = out.dropna(subset=["available_date"]).copy()
    if len(audit):
        lag = (audit["signal_date"] - audit["available_date"]).dt.days
        log.info("PIT 可得日审计：可得日期早于或等于信号日 %.1f%%；中位提前 %s 天；" 
                 "信号日后可得记录 %.1f%%",
                 100 * (lag >= 0).mean(), int(lag.median()), 100 * (lag < 0).mean())
    return out


def build_daily_ff3(daily: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """用「月内固定的分组归属」构造日频 FF3：MKT / SMB / HML。"""
    mem = membership[membership["factor"].isin(["SMB", "VAL"])][["ym", "code", "factor", "grp"]]
    d = daily.copy()
    d["ym"] = d["date"].dt.to_period("M")
    d = d.merge(mem, on=["ym", "code"], how="left", suffixes=("", "_m"))

    def _leg_ret(g: pd.DataFrame) -> float:
        g = g.dropna(subset=["ret", "float_mv"])
        if g.empty or g["float_mv"].sum() <= 0:
            return np.nan
        return float(np.average(g["ret"], weights=g["float_mv"]))

    # 只留下 _leg_ret 真正需要的列：避免 groupby.apply 把 Period 列带进返回结果
    # （空分组时 apply 会退化成返回 DataFrame，随后做 Period 相减会抛 NaTType 的 TypeError）
    d_leg = d[["date", "ret", "float_mv", "factor", "grp"]]
    mkt = d_leg.groupby("date", observed=True).apply(_leg_ret, include_groups=False).rename("MKT")
    mkt = pd.Series(mkt, dtype=float).rename("MKT")

    def _spread(sub: pd.DataFrame, hi: int, lo: int) -> pd.Series:
        def _one(g: int) -> pd.Series:
            s = sub.loc[sub["grp"] == g, ["date", "ret", "float_mv"]]
            if s.empty:
                return pd.Series(dtype=float)
            out = s.groupby("date", observed=True).apply(_leg_ret, include_groups=False)
            return pd.Series(out, dtype=float)
        return (_one(hi) - _one(lo)).rename("v")

    smb = _spread(d[d["factor"] == "SMB"], 0, N_GROUPS - 1).rename("SMB")   # 小市值 − 大市值
    hml = _spread(d[d["factor"] == "VAL"], N_GROUPS - 1, 0).rename("HML")  # 高 B/M − 低 B/M
    ff3 = pd.concat([mkt, smb, hml], axis=1).dropna()
    log.info(f"日频 FF3：{len(ff3)} 个交易日")
    return ff3


def build_return_matrix(daily: pd.DataFrame, ff3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把长表面板转成「交易日 × 股票」的收益矩阵 R，以及对齐的 FF3 因子矩阵 X。

    矩阵化是这里的关键：真实数据有 5418 只股票 × 49 个月 × 250 天 ≈ 6600 万个残差，
    逐窗口物化再 concat 会直接撑爆 8GB 机器，逐股票全表筛选又是 O(股票数 × 行数)。
    转成宽矩阵后，每个形成窗口只需一次批量最小二乘。
    """
    d = daily[["code", "date", "ret"]].merge(ff3, left_on="date", right_index=True, how="inner")
    d = d.dropna(subset=["ret", "MKT", "SMB", "HML"])
    R = d.pivot(index="date", columns="code", values="ret").astype("float32")
    X = (d.groupby("date", observed=True)[["MKT", "SMB", "HML"]].first()
         .reindex(R.index).astype("float64"))
    del d
    gc.collect()
    log.info(f"收益矩阵 {R.shape[0]} 个交易日 × {R.shape[1]} 只股票")
    return R, X


def window_residuals(R: pd.DataFrame, X: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """对窗口 [lo, hi) 内每只股票估 FF3，返回残差矩阵（行=交易日，列=股票）。

    窗口内数据完整的股票走一次批量最小二乘（同时解全部股票）；
    有停牌缺口但缺口不大的股票单独在其有效交易日上回归，不直接丢掉。
    """
    Rw = R.iloc[lo:hi]
    # 必须带截距：FF3 时序回归有常数项，漏掉会让残差带非零均值、把残差相关整体抬高
    Xw = np.column_stack([np.ones(len(Rw)), X.iloc[lo:hi].to_numpy(dtype="float64")])
    A = Rw.to_numpy(dtype="float64")
    ok = ~np.isnan(A)
    n_obs = ok.sum(axis=0)
    resid = np.full(A.shape, np.nan, dtype="float32")

    full = n_obs == A.shape[0]
    if full.any():
        cols = np.flatnonzero(full)
        beta, *_ = np.linalg.lstsq(Xw, A[:, cols], rcond=None)
        resid[:, cols] = (A[:, cols] - Xw @ beta).astype("float32")

    for c in np.flatnonzero((~full) & (n_obs >= MIN_OBS_WINDOW)):
        m = ok[:, c]
        b, *_ = np.linalg.lstsq(Xw[m], A[m, c], rcond=None)
        resid[m, c] = (A[m, c] - Xw[m] @ b).astype("float32")

    return pd.DataFrame(resid, index=Rw.index, columns=Rw.columns)


def main() -> None:
    gate = Gate(log)
    rng = np.random.default_rng(SEED)
    log.info(f"开始计算拥挤度 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    monthly = read_parquet(CLEAN / "monthly.parquet")
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    monthly["code"] = monthly["code"].astype(str)

    # 因子分组形成月 t 的信息在共同月末信号日冻结；comomentum 的形成窗口
    # 只使用截至该信号日收盘可见的日收益，下一月收益从 t+1 开始实现。
    # 两融余额另按 available_date 施加保守 T+1 可得约束。
    # ---------- 分组归属（月内固定）----------
    mem_rows = []
    for ym, g in monthly.groupby("ym", observed=True):
        # 与 06 用同一条股票池规则：06 是先剔除「当月停牌超过 10 个交易日」的观测
        # 再分十分位。08 若不过滤，拥挤度衡量的组合与主检验里真正持有并计算收益的
        # 组合就不是同一个，"用拥挤度预测该因子未来收益"这句话就打了折扣。
        g = g[g["n_days"] >= 10]
        for f in FACTORS:
            d = g.dropna(subset=[f])
            if len(d) < N_GROUPS * 5:
                continue
            try:
                d = d.assign(grp=pd.qcut(d[f].rank(method="first"), N_GROUPS, labels=False))
            except ValueError:
                continue
            mem_rows.append(d[["ym", "code", "float_mv"]].assign(factor=f, grp=d["grp"].values))
    membership = pd.concat(mem_rows, ignore_index=True)
    log.info(f"分组归属：{len(membership):,} 行")

    daily = load_daily()

    # 月末交易日必须取自**交易日历**，不能从 monthly["date"] 里收集。
    # monthly 的 date 是「每只股票各自当月的最后交易日」，真实数据里停牌股的最后交易日
    # 各不相同，于是同一个月会冒出几十个不同日期，每个都触发一次残差计算与相关系数，
    # 同一个 (月, 因子) 被算了多次、且每次窗口端点都不同。合并时这些重复行会把表炸开
    # （实测 crowding.parquet 出现 790 个重复的 (ym, factor) 组合）。
    # 合成数据没暴露这个问题，是因为合成面板里所有股票每天都有数据、每月只有一个快照日。
    cal = read_parquet(CLEAN / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    cal["ym"] = cal["date"].dt.to_period("M")
    month_ends = sorted(cal.groupby("ym", observed=True)["date"].max().tolist())
    log.info(f"月末交易日 {len(month_ends)} 个（{month_ends[0]:%Y-%m} ~ {month_ends[-1]:%Y-%m}）")
    assert len({pd.Period(x, freq="M") for x in month_ends}) == len(month_ends), \
        "月末日期出现重复月份，会再次导致 (ym, factor) 重复"

    # ---------- 测度 A：comomentum ----------
    ff3 = build_daily_ff3(daily, membership)
    R, X = build_return_matrix(daily, ff3)
    daily_dates = pd.DatetimeIndex(R.index)
    del daily, ff3
    gc.collect()
    dates = R.index.to_numpy()
    # 分组归属按 ym 预索引：否则每个月都要对全表 membership 做一次布尔筛选
    mem_by = {ym: {f: gg for f, gg in g.groupby("factor", observed=True)}
              for ym, g in membership.groupby("ym", observed=True)}

    com_rows = []
    for t in month_ends:
        if not gate.ok():
            log.warning("闸门触发")
            break
        hi = int(np.searchsorted(dates, np.datetime64(t), side="right"))
        lo = hi - FORM_WINDOW
        if lo < 0 or hi - lo < MIN_WIN_DAYS:
            continue
        ym = pd.Period(t, freq="M")
        legs = mem_by.get(ym)
        if not legs:
            continue
        # 逐月算残差、用完即弃：全部窗口的残差是 6600 万行，物化下来会 OOM
        Res = window_residuals(R, X, lo, hi)
        for f in FACTORS:
            sub = legs.get(f)
            if sub is None or sub.empty:
                continue
            val = []
            for grp in (N_GROUPS - 1, 0):
                # 多头端与空头端必须同时有效：只出一端会把单边相关当成两端差异
                codes = sub[sub["grp"] == grp]["code"].tolist()
                codes = [c for c in codes if c in Res.columns]
                if len(codes) < MIN_PER_LEG:
                    val = []
                    break
                if len(codes) > SAMPLE_PER_LEG:
                    codes = list(rng.choice(codes, size=SAMPLE_PER_LEG, replace=False))
                # 用**两两完整**观测算相关，不要把含缺失的行整行丢掉。
                # 真实数据里有停牌，100 只股票在 250 天窗口内几乎不可能全部无缺失；
                # 整行 dropna 会把除 MOM 外的所有因子都判成"算不出"
                # （MOM 幸存只是因为它的股票池本就限定在连续 12 个月有数据的股票上）。
                C = Res[codes].corr(min_periods=MIN_PAIRS).to_numpy()
                iu = np.triu_indices_from(C, k=1)
                v = C[iu]
                v = v[np.isfinite(v)]
                if v.size == 0:
                    val = []
                    break
                val.append(float(v.mean()))
            if len(val) == 2:
                com_rows.append({"ym": str(ym), "factor": f, "comomentum": float(np.mean(val)),
                                 "n_legs": len(val)})
        del Res
    del R, X, mem_by
    gc.collect()
    com = pd.DataFrame(com_rows)
    # 兜底：即使上游哪天又冒出一个月的多个端点，也不能让重复行流到合并那一步
    if len(com):
        n0 = len(com)
        com = com.drop_duplicates(["ym", "factor"], keep="first")
        if len(com) != n0:
            log.warning(f"comomentum 表有 {n0 - len(com)} 行重复的 (ym, factor)，已去重")
    log.info(f"comomentum 表：{len(com)} 行")

    # ---------- 测度 B：两融拥挤度 ----------
    mg = load_margin_monthly(daily_dates)
    crowd_rows = []
    if len(mg):
        mg["ym"] = mg["ym"].astype(str)
        base = monthly.copy()
        base["ym"] = base["ym"].astype(str)
        base = base.merge(
            mg[["code", "ym", "d_mb", "margin_balance", "available_date",
                "signal_date"]],
            on=["code", "ym"], how="left"
        )
        base["has_margin"] = base["margin_balance"].notna()
        base["intensity"] = np.where(base["float_mv"] > 0,
                                     base["d_mb"] / base["float_mv"], np.nan)
        # 两融水平：融资余额 / 流通市值。
        # 必须作为控制变量，原因见下方 _leg 的注释——「变化」里含有「水平」的机械成分。
        base["level"] = np.where(base["float_mv"] > 0,
                                 base["margin_balance"] / base["float_mv"], np.nan)

        # 与分组归属合并：membership 已按月按因子给出每只股票的十分位归属
        mm = membership.copy()
        mm["ym"] = mm["ym"].astype(str)
        bb = base.drop(columns=["float_mv"], errors="ignore")
        merged = mm.merge(bb[["code", "ym", "d_mb", "margin_balance", "has_margin",
                              "intensity", "level", "available_date", "signal_date"]],
                          on=["code", "ym"], how="left")

        for (ym, f), g in merged.groupby(["ym", "factor"], observed=True):
            if not gate.ok():
                log.warning("闸门触发")
                break
            top = g[g["grp"] == N_GROUPS - 1]
            bot = g[g["grp"] == 0]
            if len(top) < MIN_PER_LEG or len(bot) < MIN_PER_LEG:
                continue

            def _leg(x: pd.DataFrame):
                # 加权平均「融资余额变化强度」「两融水平」「覆盖率」
                w = x["float_mv"] / x["float_mv"].sum()
                v = (x["intensity"].fillna(0.0) * w).sum()
                lv = (x["level"].fillna(0.0) * w).sum()
                cov = (x["has_margin"].astype(float) * w).sum()
                return float(v), float(lv), float(cov)

            v_t, l_t, c_t = _leg(top)
            v_b, l_b, c_b = _leg(bot)
            avail = pd.concat([top["available_date"], bot["available_date"]]).dropna()
            signal = pd.concat([top["signal_date"], bot["signal_date"]]).dropna()
            crowd_rows.append({"ym": ym, "factor": f,
                               "crowd_margin": v_t - v_b,        # 多头 − 空头
                               "crowd_margin_long": v_t,          # 仅多头端
                               "level_margin": l_t - l_b,         # 两融水平差（控制变量）
                               "cover_long": c_t, "cover_short": c_b,
                               "signal_date": signal.iloc[0] if len(signal) else pd.NaT,
                               "max_available_date": avail.max() if len(avail) else pd.NaT,
                               "n_long": int(len(top)), "n_short": int(len(bot))})
    crowd = pd.DataFrame(crowd_rows)
    if len(crowd):
        n0 = len(crowd)
        crowd = crowd.drop_duplicates(["ym", "factor"], keep="first")
        if len(crowd) != n0:
            log.warning(f"两融拥挤度表有 {n0 - len(crowd)} 行重复的 (ym, factor)，已去重")

    # ---------- 合并 ----------
    out = com.merge(crowd, on=["ym", "factor"], how="outer") if len(crowd) else com
    out["ym"] = pd.PeriodIndex(out["ym"], freq="M")
    write_parquet(out.sort_values(["ym", "factor"]), CLEAN / "crowding.parquet")
    log.info(f"拥挤度表：{len(out)} 行")

    # ---------- 两个测度的相关系数（Table 1 的核心发现）----------
    rows = []
    for f in FACTORS:
        s = out[out["factor"] == f].dropna(subset=["comomentum", "crowd_margin"]) if "crowd_margin" in out else pd.DataFrame()
        if len(s) < 12:
            continue
        rows.append({"factor": f, "n_months": len(s),
                     "pearson": s["comomentum"].corr(s["crowd_margin"]),
                     "spearman": s["comomentum"].corr(s["crowd_margin"], method="spearman"),
                     "comomentum_mean": s["comomentum"].mean(),
                     "crowd_mean": s["crowd_margin"].mean()})
    pooled = out.dropna(subset=["comomentum", "crowd_margin"]) if "crowd_margin" in out else pd.DataFrame()
    if len(pooled) > 20:
        rows.append({"factor": "POOLED", "n_months": len(pooled),
                     "pearson": pooled["comomentum"].corr(pooled["crowd_margin"]),
                     "spearman": pooled["comomentum"].corr(pooled["crowd_margin"], method="spearman"),
                     "comomentum_mean": pooled["comomentum"].mean(),
                     "crowd_mean": pooled["crowd_margin"].mean()})
        # 池化相关会被「因子层面的均值差异」主导（不同因子的测度水平本就不同），
        # 因此另报一个先按因子去均值、再算相关的版本——这才是论文里该用的池化数字。
        dm = pooled.copy()
        for c in ("comomentum", "crowd_margin"):
            dm[c] = dm[c] - dm.groupby("factor", observed=True)[c].transform("mean")
        rows.append({"factor": "POOLED(因子内去均值)", "n_months": len(dm),
                     "pearson": dm["comomentum"].corr(dm["crowd_margin"]),
                     "spearman": dm["comomentum"].corr(dm["crowd_margin"], method="spearman"),
                     "comomentum_mean": dm["comomentum"].mean(),
                     "crowd_mean": dm["crowd_margin"].mean()})
    cc = pd.DataFrame(rows)
    if len(cc):
        cc.to_csv(REPORTS / "crowding_correlation.csv", index=False, encoding="utf-8-sig")
        log.info("\n★ 两个拥挤度测度的相关系数（本文核心发现）：\n" + cc.round(4).to_string())

    # ---------- 图 ----------
    # 必须用**双纵轴**：两个测度量纲差一到两个数量级（comomentum 是残差相关系数，约 0.01~0.06；
    # 两融强度约 ±0.005），画在同一根轴上会让其中一条压成直线，读者读不出任何信息。
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 7), dpi=200)
        for ax, f in zip(axes.ravel(), FACTORS):
            s = out[out["factor"] == f].sort_values("ym")
            x = s["ym"].dt.to_timestamp()
            l1, = ax.plot(x, s["comomentum"].rolling(3, min_periods=1).mean(),
                          color="#1f77b4", lw=1.5, label="Comomentum (left axis)")
            for ev in ("2015-06-01", "2024-09-01"):
                ax.axvline(pd.Timestamp(ev), color="grey", ls=":", lw=1)
            ax.set_title(f)
            ax.set_ylabel("comomentum", color="#1f77b4", fontsize=8)
            ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(alpha=.3)
            if "crowd_margin" in s.columns:
                ax2 = ax.twinx()
                l2, = ax2.plot(x, s["crowd_margin"].rolling(3, min_periods=1).mean(),
                               color="#d62728", lw=1.5, label="Margin crowding (right axis)")
                ax2.set_ylabel("margin crowding", color="#d62728", fontsize=8)
                ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=7)
                if f == FACTORS[0]:
                    ax.legend(handles=[l1, l2], fontsize=7, loc="best")
        # 不设总标题：期刊图的标题由 LaTeX 的 \caption 负责，图里再写一遍是重复。
        fig.tight_layout()
        fig.savefig(REPORTS / "fig_crowding_series.png", bbox_inches="tight")
        log.info("图已保存 fig_crowding_series.png（双纵轴）")
    except Exception as e:  # noqa: BLE001
        log.warning(f"绘图失败：{e}")

    heartbeat("08 拥挤度测度结束", log)


if __name__ == "__main__":
    main()
