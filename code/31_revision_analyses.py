"""
31_revision_analyses.py —— 2026-10-25 复审补充分析

回应复审意见中的四组问题，全部沿用主分析的面板、控制变量与推断口径：

1. 稳健性表统一到规格 D（A2/M6）：子样本、等权、6/24 个月余额变化窗口、剔除 SMB、
   覆盖率门槛、复利收益、90 天陈旧阈值，h ∈ {1, 6}，均为规格 D + 同一循环平移置换。
2. 因子特定斜率（B3）：在规格 D 的 pooled 回归中加入测度 × 因子交互，报告六个斜率、
   斜率相等的 HAC Wald 检验，以及逐因子剔除的 pooled 系数。
3. 置换敏感性（B4）：剔除 min(k, n−k) ≤ 12 的近邻位移后重算主表规格 D 的 p 值，
   检验 comomentum 与融资代理的推断差异是否来自两者持续性不同。
4. 机制（B1）：股票层面 Fama–MacBeth。在空头腿与多头腿成分股中，分别检验
   股票自身融资强度是否预测下月收益，并检验是否集中在彩票型（高 MAX）股票。

产出：
  reports/rev_robustness_specD.csv
  reports/rev_factor_slopes.csv
  reports/rev_permutation_sensitivity.csv
  reports/rev_stock_mechanism.csv
  reports/rev_family_maxT.csv      主检验族（2 测度 × h∈{1,3,6}）的 max-|t| 联合置换校正
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from common import (CLEAN, REPORTS, _grouped_circular_shift, driscoll_kraay_cov,  # noqa: E402
                    formation_universe, get_logger, hac_ols, now_local, read_parquet,
                    time_index)

log = get_logger("31_revision", "31_revision_analyses.log")
m08 = importlib.import_module("08_crowding")
m09 = importlib.import_module("09_regressions")
m16 = importlib.import_module("16_extension_analyses")
m24 = importlib.import_module("24_margin_staleness_robustness")

FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
HS = [1, 6]


def fe_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("FE_")]


# ------------------------------------------------------------------
# 1) 规格 D 稳健性
# ------------------------------------------------------------------
def equal_weight_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """与 11_robustness 相同的等权十分位多空收益（形成月成员、下月实现）。"""
    rows = []
    for ym, g in monthly.groupby("ym", observed=True):
        base = formation_universe(g)
        if len(base) < m08.N_GROUPS * 5:
            continue
        for f in FACTORS:
            d = base.dropna(subset=[f])
            if len(d) < m08.N_GROUPS * 5:
                continue
            d = d.assign(grp=pd.qcut(d[f].rank(method="first"), m08.N_GROUPS, labels=False))
            top = d[d["grp"] == m08.N_GROUPS - 1]
            bot = d[d["grp"] == 0]
            rows.append({"ym": ym, "factor": f,
                         "ls_ew": top["ret_next"].mean() - bot["ret_next"].mean()})
    eq = pd.DataFrame(rows).sort_values(["factor", "ym"])
    g = eq.groupby("factor", observed=True)["ls_ew"]
    for k in (1, 6, 12):
        eq[f"past{k}_ew"] = g.transform(lambda s: s.shift(1).rolling(k, min_periods=k).sum())
    for h in HS:
        eq[f"fwd{h}_ew"] = g.transform(lambda s: s.rolling(h, min_periods=h).sum().shift(-(h - 1)))
    return eq


def robustness_specD(df: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    ctrl = m16.controls_for(df, "D")
    rows = []

    def add(name, sub, measure, y, h, cols=None):
        r = m16.fit_row(sub, cols or ([measure] + ctrl), measure, y,
                        {"check": name, "horizon": h})
        if r:
            rows.append(r)

    for h in HS:
        y = f"fwd{h}"
        for meas in ["comomentum_z", "crowd_margin_z"]:
            add("Baseline (specification D)", df, meas, y, h)
            for name, a, b in [("Subsample 2015--2018", "2015-01", "2018-12"),
                               ("Subsample 2019--2021", "2019-01", "2021-12"),
                               ("Subsample 2022--2025", "2022-01", "2025-12")]:
                sub = df[(df["ym"] >= pd.Period(a, "M")) & (df["ym"] <= pd.Period(b, "M"))]
                add(name, sub, meas, y, h)
            add("Excluding size (SMB)", df[df["factor"] != "SMB"], meas, y, h)
            add("Long-leg coverage $\\geq$ 0.50", df[df["cover_long"] >= 0.50], meas, y, h)
            add("Compounded returns", df, meas, f"{y}_c", h)

    # 等权：收益与过去收益控制都换成等权序列
    eq = equal_weight_returns(monthly)
    dfe = df.merge(eq, on=["ym", "factor"], how="inner")
    ctrl_ew = (fe_cols(dfe) + ["past1_ew", "past6_ew", "past12_ew", "level_margin_z",
                               "cover_long_z", "cover_short_z"])
    for h in HS:
        for meas in ["comomentum_z", "crowd_margin_z"]:
            add("Equal-weighted legs", dfe, meas, f"fwd{h}_ew", h, [meas] + ctrl_ew)

    # 6/24 个月余额变化窗口与 90 天陈旧阈值：只替换融资测度构造
    cal = pd.DatetimeIndex(pd.to_datetime(
        read_parquet(CLEAN / "calendar.parquet")["date"])).sort_values().unique()
    membership = m24.build_membership(monthly)
    base = m09.build_panel()
    for h in HS:
        base[f"fwd{h}_c"] = df.set_index(["ym", "factor"])[f"fwd{h}_c"].reindex(
            pd.MultiIndex.from_frame(base[["ym", "factor"]])).to_numpy()
    mg45 = m08.load_margin_monthly(cal, max_stale_days=45)
    variants = {}
    for w in (6, 24):
        mg_w = mg45.copy()
        lag = mg45[["code", "ym", "margin_balance"]].copy()
        lag["ym"] = lag["ym"] + w
        lag = lag.rename(columns={"margin_balance": "mb_lag"})
        mg_w = mg_w.merge(lag, on=["code", "ym"], how="left", validate="one_to_one")
        mg_w["d_mb"] = mg_w["margin_balance"] - mg_w["mb_lag"]
        variants[f"{w}-month balance change"] = m08.build_margin_crowding(monthly, membership, mg_w)
    mg90 = m08.load_margin_monthly(cal, max_stale_days=90)
    variants["90-day staleness cutoff"] = m08.build_margin_crowding(monthly, membership, mg90)
    for name, cw in variants.items():
        panel = m24.variant_panel(base, cw)
        ctrl_v = fe_cols(panel) + m09.MOM_CTRL + ["level_margin_z", "cover_long_z", "cover_short_z"]
        for h in HS:
            add(name, panel, "crowd_margin_z", f"fwd{h}", h, ["crowd_margin_z"] + ctrl_v)

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "rev_robustness_specD.csv", index=False, encoding="utf-8-sig")
    log.info("规格 D 稳健性：\n" + out[["check", "measure", "horizon", "beta", "t",
                                       "p_placebo", "n"]].round(4).to_string())
    return out


# ------------------------------------------------------------------
# 2) 因子特定斜率
# ------------------------------------------------------------------
def _interaction_fit(d: pd.DataFrame, measure: str, ctrl: list[str], y: str, lags: int,
                     time=None):
    X = d[ctrl].copy()
    names = []
    for f in FACTORS:
        c = f"{measure}_x_{f}"
        X[c] = d[measure].to_numpy() * (d["factor"] == f).to_numpy(dtype=float)
        names.append(c)
    cols = names + ctrl
    import statsmodels.api as sm
    Xc = sm.add_constant(X[cols].to_numpy(dtype=float), has_constant="add")
    if time is None:
        res = sm.OLS(d[y].to_numpy(dtype=float), Xc).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags})
        cov = res.cov_params()
    else:
        # 同样是因子 × 月堆叠面板：交互项回归也必须用 DK，否则斜率与 Wald
        # 的 t 值仍在按行计算滞后。
        res = sm.OLS(d[y].to_numpy(dtype=float), Xc).fit()
        cov = driscoll_kraay_cov(Xc, res.resid, time, lags)
    b = res.params[1:1 + len(FACTORS)]
    V = cov[1:1 + len(FACTORS), 1:1 + len(FACTORS)]
    t = b / np.sqrt(np.diag(V))
    R = np.hstack([np.ones((len(FACTORS) - 1, 1)), -np.eye(len(FACTORS) - 1)])
    rb = R @ b
    wald = float(rb @ np.linalg.pinv(R @ V @ R.T) @ rb)
    return b, t, wald


def factor_slopes(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import chi2
    ctrl = m16.controls_for(df, "D")
    rows = []
    for meas in ["comomentum_z", "crowd_margin_z"]:
        for h in HS:
            y = f"fwd{h}"
            lags = 6
            d = df.dropna(subset=[meas] + ctrl + [y]).sort_values(["ym", "factor"]).copy()
            tvals = time_index(d["ym"])          # A 类堆叠面板 → DK
            b, t, wald = _interaction_fit(d, meas, ctrl, y, lags, time=tvals)
            n_sh = d.groupby("factor").size().min() - 1
            null_t, null_w = [], []
            for k in range(1, n_sh + 1):
                p = _grouped_circular_shift(d, meas, "factor", k)
                bb, tt, ww = _interaction_fit(p, meas, ctrl, y, lags, time=tvals)
                null_t.append(np.abs(tt))
                null_w.append(ww)
            null_t = np.array(null_t)
            null_w = np.array(null_w)
            for i, f in enumerate(FACTORS):
                pp = (1 + np.sum(null_t[:, i] >= abs(t[i]))) / (1 + len(null_t))
                rows.append({"measure": meas.replace("_z", ""), "horizon": h, "analysis": "slope",
                             "factor": f, "beta": b[i], "t": t[i], "p_placebo": pp,
                             "n": int((d["factor"] == f).sum())})
            rows.append({"measure": meas.replace("_z", ""), "horizon": h,
                         "analysis": "equality Wald", "factor": "all",
                         "beta": np.nan, "t": wald,
                         "p": float(chi2.sf(wald, len(FACTORS) - 1)),
                         "p_placebo": (1 + np.sum(null_w >= wald)) / (1 + len(null_w)),
                         "n": len(d)})
            for f in FACTORS:
                r = m16.fit_row(df[df["factor"] != f], [meas] + ctrl, meas, y,
                                {"analysis": "leave one out", "factor": f, "horizon": h})
                if r:
                    rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "rev_factor_slopes.csv", index=False, encoding="utf-8-sig")
    log.info("因子特定斜率：\n" + out[["measure", "horizon", "analysis", "factor", "beta",
                                      "t", "p_placebo"]].round(4).to_string())
    return out


# ------------------------------------------------------------------
# 3) 置换敏感性：剔除近邻位移
# ------------------------------------------------------------------
def permutation_sensitivity(df: pd.DataFrame, exclude: int = 12) -> pd.DataFrame:
    ctrl = m16.controls_for(df, "D")
    rows = []
    for meas in ["comomentum_z", "crowd_margin_z"]:
        for h in [1, 3, 6]:
            y = f"fwd{h}"
            cols = [meas] + ctrl
            d = df.dropna(subset=cols + [y]).sort_values(["ym", "factor"]).copy()
            tvals = time_index(d["ym"])          # A 类堆叠面板 → DK
            r = hac_ols(d[y].to_numpy(), d[cols].to_numpy(), cols, lags=6, time=tvals)
            t_obs = abs(r[meas]["t"])
            n_min = d.groupby("factor").size().min()
            ts_all, ts_far = [], []
            for k in range(1, n_min):
                p = _grouped_circular_shift(d, meas, "factor", k)
                rr = hac_ols(p[y].to_numpy(), p[cols].to_numpy(), cols, lags=6, time=tvals)
                tk = abs(rr[meas]["t"])
                ts_all.append(tk)
                if min(k, n_min - k) > exclude:
                    ts_far.append(tk)
            ts_all, ts_far = np.array(ts_all), np.array(ts_far)
            rows.append({"measure": meas.replace("_z", ""), "horizon": h,
                         "t_obs": r[meas]["t"], "beta": r[meas]["coef"],
                         "B_all": len(ts_all),
                         "p_all": (1 + np.sum(ts_all >= t_obs)) / (1 + len(ts_all)),
                         "crit95_all": float(np.quantile(ts_all, 0.95)),
                         "B_far": len(ts_far),
                         "p_far": (1 + np.sum(ts_far >= t_obs)) / (1 + len(ts_far)),
                         "crit95_far": float(np.quantile(ts_far, 0.95)),
                         "min_p_all": 1 / (1 + len(ts_all))})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "rev_permutation_sensitivity.csv", index=False, encoding="utf-8-sig")
    log.info("置换敏感性：\n" + out.round(4).to_string())
    return out


# ------------------------------------------------------------------
# 3b) 主检验族的联合置换校正（max-|t|，Westfall–Young 逐步下降）
# ------------------------------------------------------------------
FAMILY_MEASURES = ["comomentum_z", "crowd_margin_z"]
FAMILY_HORIZONS = [1, 3, 6]


def family_maxT(df: pd.DataFrame) -> pd.DataFrame:
    """六个主检验（2 测度 × h∈{1,3,6}）的 max-|t| 联合置换校正。

    观测与置换统计量都用规格 D 控制变量、DK 标准误、L=6。置换时**两个测度
    同时**按因子循环平移同一个 k，这样零分布保留了两测度之间的同期相关；
    每个检验用各自 dropna 后的样本，平移在该样本内按因子进行。
    """
    ctrl = m16.controls_for(df, "D")
    tests = []
    for measure in FAMILY_MEASURES:
        for h in FAMILY_HORIZONS:
            y = f"fwd{h}"
            cols = [measure] + ctrl
            d = (df.dropna(subset=cols + [y])
                  .sort_values(["ym", "factor"]).reset_index(drop=True))
            if len(d) < 40:
                raise ValueError(f"family 检验样本过小：{measure} h={h}, N={len(d)}")
            tvals = time_index(d["ym"])
            r = hac_ols(d[y].to_numpy(dtype=float), d[cols].to_numpy(dtype=float), cols,
                        lags=6, time=tvals)
            tests.append({"measure": measure, "horizon": h, "y": y, "d": d,
                          "cols": cols, "t": float(r[measure]["t"])})
    # 所有检验共用一个位移集合：取各检验「最短因子序列 − 1」的最小值。
    K = min(int(t["d"].groupby("factor", observed=True).size().min()) - 1 for t in tests)
    if K < 1:
        raise ValueError("family 置换位移数不足")
    log.info(f"family：{len(tests)} 个检验，共同位移 K={K}")

    null = np.empty((K, len(tests)), dtype=float)
    for k in range(1, K + 1):
        for j, t in enumerate(tests):
            perm = _grouped_circular_shift(t["d"], t["measure"], "factor", k)
            tvals = time_index(perm["ym"])
            rr = hac_ols(perm[t["y"]].to_numpy(dtype=float),
                         perm[t["cols"]].to_numpy(dtype=float), t["cols"], lags=6, time=tvals)
            null[k - 1, j] = abs(float(rr[t["measure"]]["t"]))

    obs = np.abs(np.array([t["t"] for t in tests], dtype=float))
    p_unadj = (1.0 + np.sum(null >= obs[None, :], axis=0)) / (1.0 + K)
    max_stat = null.max(axis=1)
    p_maxT_single = (1.0 + np.sum(max_stat[:, None] >= obs[None, :], axis=0)) / (1.0 + K)

    # Westfall–Young 逐步下降：按 |t| 从大到小，每一步的 max 只在剩余检验里取；
    # 最后对 p 做单调化，保证排在前面（更强）的检验不会得到更大的校正 p。
    order = np.argsort(-obs, kind="stable")
    p_step = np.empty(len(tests), dtype=float)
    prev = 0.0
    for rank, j in enumerate(order):
        remaining = order[rank:]
        m_r = null[:, remaining].max(axis=1)
        pj = (1.0 + np.sum(m_r >= obs[j])) / (1.0 + K)
        prev = max(prev, pj)
        p_step[j] = prev

    out = pd.DataFrame({
        "measure": [t["measure"].replace("_z", "") for t in tests],
        "horizon": [t["horizon"] for t in tests],
        "t": [t["t"] for t in tests],
        "p_unadj": p_unadj,
        "p_maxT_single": p_maxT_single,
        "p_maxT_stepdown": p_step,
        "K": K,
    })
    out.to_csv(REPORTS / "rev_family_maxT.csv", index=False, encoding="utf-8-sig")
    log.info("主检验族 max-|t| 校正：\n" + out.round(4).to_string(index=False))
    return out


# ------------------------------------------------------------------
# 4) 股票层面机制：腿内融资强度 → 下月收益
# ------------------------------------------------------------------
def monthly_max() -> pd.DataFrame:
    files = sorted((CLEAN / "panel").rglob("*.parquet"))
    parts = []
    for f in files:
        x = pd.read_parquet(f, columns=["date", "code", "ret"])
        x["ym"] = pd.to_datetime(x["date"]).dt.to_period("M")
        parts.append(x.groupby(["code", "ym"], observed=True)["ret"].max().rename("MAX").reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["code"] = out["code"].astype(str)
    return out


def _nw_mean(x: np.ndarray, lags: int = 6) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    m = x.mean()
    e = x - m
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    return float(m), float(m / np.sqrt(s / n))


def stock_mechanism(monthly: pd.DataFrame) -> pd.DataFrame:
    cal = pd.DatetimeIndex(pd.to_datetime(
        read_parquet(CLEAN / "calendar.parquet")["date"])).sort_values().unique()
    mg = m08.load_margin_monthly(cal, max_stale_days=45)
    mg["ym"] = pd.PeriodIndex(mg["ym"], freq="M")
    membership = m24.build_membership(monthly)
    membership["ym"] = pd.PeriodIndex(membership["ym"], freq="M")
    mx = monthly_max()

    st = monthly[["code", "ym", "float_mv", "mret", "ret_next"]].copy()
    st["code"] = st["code"].astype(str)
    st = st.merge(mg[["code", "ym", "d_mb", "margin_balance"]], on=["code", "ym"], how="left")
    st = st.merge(mx, on=["code", "ym"], how="left")
    st["intensity"] = st["d_mb"] / st["float_mv"]
    st["log_mv"] = np.log(st["float_mv"])
    # 只用融资余额有效的两融标的：机制检验比较的是标的之间的融资差异
    st = st.dropna(subset=["intensity", "ret_next", "MAX", "log_mv", "mret"])
    # 全截面（两融标的）月度 MAX 中位数，用于比较两腿的彩票型股票占比
    st["high_mkt"] = (st["MAX"] > st.groupby("ym", observed=True)["MAX"].transform("median")
                      ).astype(float)

    legs = {"Short leg": membership[membership["grp"] == 0],
            "Long leg": membership[membership["grp"] == m08.N_GROUPS - 1]}
    rows = []
    for leg_name, mem in legs.items():
        codes = mem[["code", "ym"]].drop_duplicates()
        codes["code"] = codes["code"].astype(str)
        s = st.merge(codes, on=["code", "ym"], how="inner")
        coefs = {k: [] for k in ["intensity", "intensity_x_highMAX", "intensity_only"]}
        nobs = []
        for ym, g in s.groupby("ym", observed=True):
            if len(g) < 50:
                continue
            g = g.copy()
            lo, hi = g["intensity"].quantile([0.01, 0.99])
            g["intensity"] = g["intensity"].clip(lo, hi)
            g["iz"] = (g["intensity"] - g["intensity"].mean()) / g["intensity"].std()
            g["high"] = (g["MAX"] > g["MAX"].median()).astype(float)
            g["iz_high"] = g["iz"] * g["high"]
            for c in ["log_mv", "mret", "MAX"]:
                g[c + "_z"] = (g[c] - g[c].mean()) / g[c].std()
            y = g["ret_next"].to_numpy(dtype=float)
            X1 = np.column_stack([np.ones(len(g)), g[["iz", "log_mv_z", "mret_z", "MAX_z"]]])
            b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
            X2 = np.column_stack([np.ones(len(g)),
                                  g[["iz", "iz_high", "high", "log_mv_z", "mret_z", "MAX_z"]]])
            b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
            coefs["intensity_only"].append(b1[1])
            coefs["intensity"].append(b2[1])
            coefs["intensity_x_highMAX"].append(b2[2])
            nobs.append(len(g))
        for term, label in [("intensity_only", "Financing intensity (no interaction)"),
                            ("intensity", "Financing intensity, low-MAX stocks"),
                            ("intensity_x_highMAX", "Financing intensity x high MAX")]:
            m, t = _nw_mean(np.array(coefs[term]))
            rows.append({"leg": leg_name, "term": label, "coef": m, "t_nw": t,
                         "months": len(coefs[term]), "avg_stocks": float(np.mean(nobs))})
        # 高 MAX 组的总效应
        tot = np.array(coefs["intensity"]) + np.array(coefs["intensity_x_highMAX"])
        m, t = _nw_mean(tot)
        rows.append({"leg": leg_name, "term": "Financing intensity, high-MAX stocks",
                     "coef": m, "t_nw": t, "months": len(tot),
                     "avg_stocks": float(np.mean(nobs))})
        share = s.groupby("ym", observed=True)["high_mkt"].mean()
        rows.append({"leg": leg_name, "term": "Share of stock-months above market median MAX",
                     "coef": float(share.mean()), "t_nw": np.nan, "months": len(share),
                     "avg_stocks": float(np.mean(nobs))})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "rev_stock_mechanism.csv", index=False, encoding="utf-8-sig")
    log.info("股票层面机制：\n" + out.round(5).to_string())
    return out


FACTOR_NAMES = {"VAL": "Value", "SMB": "Size", "MOM": "Momentum", "QMJ": "Quality",
                "LOWVOL": "Low volatility", "TURN": "Low turnover"}


def figure_series() -> None:
    """Figure 1：两测度的因子内标准化序列（3 个月滚动均值），单纵轴、矢量 PDF。

    原图用双纵轴且量纲不同，读者无法比较；改为因子内 z 分数后可在同一坐标上
    直接看两者是否同步。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cr = read_parquet(CLEAN / "crowding.parquet")
    cr["ym"] = pd.PeriodIndex(cr["ym"], freq="M")
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharex=True, sharey=True)
    for ax, f in zip(axes.ravel(), FACTORS):
        s = cr[cr["factor"] == f].sort_values("ym").set_index("ym")
        x = s.index.to_timestamp()
        for col, color, ls, lab in [("comomentum", "#1f4e9c", "-", "Comomentum"),
                                    ("crowd_margin", "#c0392b", "--", "Margin-balance proxy")]:
            z = (s[col] - s[col].mean()) / s[col].std()
            ax.plot(x, z.rolling(3, min_periods=3).mean(), color=color, ls=ls, lw=1.0, label=lab)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(FACTOR_NAMES[f], fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7.5)
    axes[0, 0].set_ylabel("Within-factor z-score")
    axes[1, 0].set_ylabel("Within-factor z-score")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = CODE.parent / "paper" / "figures" / "fig_crowding_series.pdf"
    fig.savefig(out)
    log.info(f"写出 {out}")


def main() -> None:
    log.info(f"31 复审补充分析 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")
    which = set(sys.argv[1:]) or {"robust", "slopes", "perm", "family", "mech", "fig"}
    if "fig" in which:
        figure_series()
        if which == {"fig"}:
            return
    df = m16.build_panel()
    monthly = read_parquet(CLEAN / "monthly.parquet")
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    if "perm" in which:
        permutation_sensitivity(df)
    if "family" in which:
        family_maxT(df)
    if "slopes" in which:
        factor_slopes(df)
    if "mech" in which:
        stock_mechanism(monthly)
    if "robust" in which:
        robustness_specD(df, monthly)


if __name__ == "__main__":
    main()
