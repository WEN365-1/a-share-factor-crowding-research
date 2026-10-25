"""
11_robustness.py —— 稳健性（Table 5）

精简版只做三项，与 01-Research-Proposal 5.4 节一致：
  1. 子样本：2015–2018 / 2019–2021 / 2022–2025（对应 A 股投资者结构三段变迁）
  2. 等权 vs 流通市值加权组合，重算因子收益与主回归
  3. 更换拥挤度的构造窗口（形成期 6 / 24 个月）

产出：reports/table5_robustness.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, REPORTS, formation_universe, get_logger, hac_ols,  # noqa: E402
                    heartbeat, now_local, placebo_pvalue, read_parquet, regression_frame,
                    time_index)

log = get_logger("11_robust", "11_robustness.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
N_GROUPS = 10


def fit(d: pd.DataFrame, measure: str, y: str = "fwd1"):
    """HAC 回归 + 循环平移安慰剂 p 值。**一律含因子固定效应。**

    为什么稳健性表也必须加因子固定效应：它不是可选的规格，而是对一处机械污染的修正。
    6 个因子的拥挤度水平与收益水平本就不同，不加固定效应，系数会有一部分来自
    「哪个因子整体更拥挤、整体收益更高」——这正是截面双变量排序里踩过的坑。
    主检验（09）已经加了；稳健性表若不加，两者不可比，还会把那处偏差当成「稳健」。

    另外，论文口径是「以安慰剂 p 为准」，所以稳健性表也必须一并报出——
    只在一处报、别处只报 HAC 的 p，读者会以为那些数字可以直接相互比较。
    """
    # 09.build_panel() 已含 FE 列；重建固定效应前先移除，避免同名列重复进入设计矩阵。
    base = d.drop(columns=[c for c in d.columns if c.startswith("FE_")], errors="ignore")
    fe = pd.get_dummies(base["factor"], prefix="FE", drop_first=True).astype(float)
    dd = regression_frame(pd.concat([base, fe], axis=1))
    cols = [measure] + list(fe.columns)
    r = hac_ols(dd[y].to_numpy(), dd[cols].to_numpy(), cols, time=time_index(dd["ym"]))
    pp, _ = placebo_pvalue(dd, cols, measure, y, r[measure]["t"])
    return r[measure]["coef"], r[measure]["t"], r[measure]["p"], pp, r["_n"]


def main() -> None:
    fr = read_parquet(CLEAN / "factor_returns.parquet")
    fr["date"] = pd.to_datetime(fr["date"])
    fr["ym"] = fr["date"].dt.to_period("M")
    cr = read_parquet(CLEAN / "crowding.parquet")
    cr["ym"] = pd.PeriodIndex(cr["ym"], freq="M")
    df = fr.merge(cr, on=["ym", "factor"], how="inner").sort_values(["factor", "date"])
    # ls_fwd1 在 t 时点记录的就是 t+1 月实现的收益，即它本身就是「未来 1 个月收益」，
    # 不需要再 shift(-1)（shift 一次会跳到 t+2 月）。
    df["fwd1"] = df["ls_fwd1"]
    for c in ["comomentum", "crowd_margin", "crowd_margin_long"]:
        if c in df.columns:
            df[c] = (df[c] - df[c].mean()) / df[c].std()
    log.info(f"稳健性面板 {len(df)} 行 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    rows = []

    # ---------- 1. 子样本 ----------
    segs = [("2015–2018 杠杆牛熊", "2015-01", "2018-12"),
            ("2019–2021 北向+主动基金共振", "2019-01", "2021-12"),
            ("2022–2025 两融散户主导", "2022-01", "2025-12")]
    for name, a, b in segs:
        a = pd.Period(a, freq="M")
        b = pd.Period(b, freq="M")
        s = df[(df["ym"] >= a) & (df["ym"] <= b)]
        for measure in ["comomentum", "crowd_margin"]:
            if measure not in s.columns:
                continue
            d = s.dropna(subset=[measure, "fwd1"])
            if len(d) < 24:
                continue
            beta, t, p, pp, n = fit(d, measure)
            rows.append({"检验": "子样本", "分组": name, "测度": measure,
                         "beta": beta, "t": t, "p": p, "p_placebo": pp, "n": n})

    # ---------- 2. 等权 vs 市值加权 ----------
    monthly = read_parquet(CLEAN / "monthly.parquet")
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    eq_rows = []
    for ym, g in monthly.groupby("ym", observed=True):
        base = formation_universe(g)
        if len(base) < N_GROUPS * 5:
            continue
        for f in FACTORS:
            d = base.dropna(subset=[f])
            if len(d) < N_GROUPS * 5:
                continue
            try:
                d = d.assign(grp=pd.qcut(d[f].rank(method="first"), N_GROUPS, labels=False))
            except ValueError:
                continue
            top = d[d["grp"] == N_GROUPS - 1]
            bot = d[d["grp"] == 0]
            eq_rows.append({"ym": ym, "factor": f,
                            "ls_ew_fwd1": top["ret_next"].mean() - bot["ret_next"].mean()})
    eq = pd.DataFrame(eq_rows)
    if len(eq):
        eq = eq.sort_values(["factor", "ym"])
        eq["fwd1_ew"] = eq["ls_ew_fwd1"]
        mg2 = df.merge(eq[["ym", "factor", "fwd1_ew"]], on=["ym", "factor"], how="inner")
        for measure in ["comomentum", "crowd_margin"]:
            if measure not in mg2.columns:
                continue
            d = mg2.dropna(subset=[measure, "fwd1_ew"])
            if len(d) < 40:
                continue
            beta, t, p, pp, n = fit(d, measure, y="fwd1_ew")
            rows.append({"检验": "等权组合", "分组": "等权（不按市值加权）", "测度": measure,
                         "beta": beta, "t": t, "p": p, "p_placebo": pp, "n": n})

    # ---------- 3. 更换拥挤度构造窗口 ----------
    # 只替换测度构造，保持 Table 5 的「测度 + 因子固定效应」估计量不变。
    # 使用主分析的 T+1、45 天陈旧阈值、腿内有效观测权重与双覆盖率规则；
    # 6/24 个月差分按自然月精确对齐，端点缺失保持缺失，不跨缺口、不填零。
    import importlib
    m08 = importlib.import_module("08_crowding")
    m09 = importlib.import_module("09_regressions")
    m24 = importlib.import_module("24_margin_staleness_robustness")
    cal = pd.DatetimeIndex(pd.to_datetime(
        read_parquet(CLEAN / "calendar.parquet")["date"])).sort_values().unique()
    mg45 = m08.load_margin_monthly(cal, max_stale_days=45)
    membership = m24.build_membership(monthly)
    base = m09.build_panel()
    for w in (6, 24):
        mg_w = mg45.copy()
        lag = mg45[["code", "ym", "margin_balance"]].copy()
        lag["ym"] = lag["ym"] + w
        lag = lag.rename(columns={"margin_balance": "mb_lag"})
        mg_w = mg_w.merge(lag, on=["code", "ym"], how="left", validate="one_to_one")
        mg_w["d_mb"] = mg_w["margin_balance"] - mg_w["mb_lag"]
        cw = m08.build_margin_crowding(monthly, membership, mg_w)
        panel = m24.variant_panel(base, cw)
        d = panel.dropna(subset=["crowd_margin_z", "fwd1"])
        if len(d) >= 40:
            beta, t, p, pp, n = fit(d, "crowd_margin_z")
            rows.append({"检验": "构造窗口", "分组": f"两融余额 {w} 个月变化", "测度": "crowd_margin",
                         "beta": beta, "t": t, "p": p, "p_placebo": pp, "n": n})

    # ---------- 4. 测度变体：仅多头端的两融拥挤度 ----------
    # 制度依据：中国的两融是极度不对称的（融资余额远大于融券，是事实上的「单边市」），
    # 而 Bian et al. (2026) 用账户级数据证明两融账户**只对负面冲击抛售、
    # 对正面冲击不显著加仓**。所以「多头端的杠杆资金是否挤进来」与
    # 「多空两端之差」未必是同一件事，值得单列。
    if "crowd_margin_long" in df.columns:
        d = df.dropna(subset=["crowd_margin_long", "fwd1"])
        if len(d) >= 40:
            beta, t, p, pp, n = fit(d, "crowd_margin_long")
            rows.append({"检验": "测度变体", "分组": "仅多头端（中国两融高度不对称）",
                         "测度": "crowd_margin_long",
                         "beta": beta, "t": t, "p": p, "p_placebo": pp, "n": n})
        # 对称检验：仅空头端。crowd_margin = v_long - v_short，故 v_short = v_long - crowd_margin。
        df["crowd_margin_short"] = df["crowd_margin_long"] - df["crowd_margin"]
        d = df.dropna(subset=["crowd_margin_short", "fwd1"])
        if len(d) >= 40:
            beta, t, p, pp, n = fit(d, "crowd_margin_short")
            rows.append({"检验": "测度变体", "分组": "仅空头端",
                         "测度": "crowd_margin_short",
                         "beta": beta, "t": t, "p": p, "p_placebo": pp, "n": n})

    t5 = pd.DataFrame(rows)
    t5.to_csv(REPORTS / "table5_robustness.csv", index=False, encoding="utf-8-sig")
    log.info("\n★ Table 5 稳健性：\n" + t5.round(4).to_string())
    heartbeat("11 稳健性结束", log)


if __name__ == "__main__":
    main()
