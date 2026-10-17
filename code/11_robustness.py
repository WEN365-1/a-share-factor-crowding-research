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
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/金融科研1014/code")
from common import (CLEAN, MARGIN, REPORTS, get_logger, hac_ols, heartbeat, now_local,  # noqa: E402
                    placebo_pvalue, read_parquet)

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
    fe = pd.get_dummies(d["factor"], prefix="FE", drop_first=True).astype(float)
    dd = pd.concat([d, fe], axis=1)
    cols = [measure] + list(fe.columns)
    r = hac_ols(dd[y].to_numpy(), dd[cols].to_numpy(), cols)
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
        base = g.dropna(subset=["ret_next", "float_mv"])
        base = base[base["float_mv"] > 0]
        # 与 06 同一条股票池规则，否则等权组合与市值加权组合不是在同一批股票上比
        base = base[base["n_days"] >= 10]
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
    # 两融拥挤度：把「月度变化」换成「6 个月 / 24 个月累计变化」
    files = sorted(MARGIN.rglob("*.parquet"))
    parts = []
    for fp in files:
        try:
            x = pd.read_parquet(fp, columns=["date", "code", "margin_balance"])
        except Exception:  # noqa: BLE001
            continue
        if len(x):
            parts.append(x)
    if parts:
        mg = pd.concat(parts, ignore_index=True)
        mg["date"] = pd.to_datetime(mg["date"])
        mg["code"] = mg["code"].astype(str)
        cal = pd.DatetimeIndex(pd.to_datetime(
            read_parquet(CLEAN / "calendar.parquet")["date"]
        )).sort_values().unique()
        next_day = pd.Series(cal[1:], index=cal[:-1])
        mg["available_date"] = mg["date"].map(next_day)
        mg = mg.dropna(subset=["available_date"])
        mg["ym"] = mg["date"].dt.to_period("M")
        last = mg.groupby(["code", "ym"], observed=True)["available_date"].idxmax()
        me = mg.loc[last, ["code", "ym", "available_date", "margin_balance"]].sort_values(["code", "ym"])
        monthly2 = monthly[["ym", "code", "float_mv", "n_days"] + FACTORS].copy()
        # 同样补上停牌剔除，与 06 的十分位口径一致
        monthly2 = monthly2[monthly2["n_days"] >= 10].copy()
        monthly2["ym"] = monthly2["ym"].astype(str)
        me["ym"] = me["ym"].astype(str)
        mm = monthly2.merge(me, on=["code", "ym"], how="left")
        mm = mm.sort_values(["code", "ym"])
        # 差分窗口按自然月位置对齐；缺失月份不会被误当作相邻观测。
        mm["ym_period"] = pd.PeriodIndex(mm["ym"], freq="M")
        for w in (6, 24):
            target = mm[["code", "ym_period", "margin_balance"]].copy()
            target["ym_period"] = target["ym_period"] + w
            target = target.rename(columns={"margin_balance": f"mb_lag{w}"})
            mm = mm.merge(target[["code", "ym_period", f"mb_lag{w}"]],
                          on=["code", "ym_period"], how="left")
            mm[f"d{w}"] = mm["margin_balance"] - mm[f"mb_lag{w}"]

        # 重新构造分组归属（与 08 一致的口径）
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
        mgm = mem.merge(mm[["code", "ym", "d6", "d24"]], on=["code", "ym"], how="left")
        mgm["int6"] = np.where(mgm["float_mv"] > 0, mgm["d6"] / mgm["float_mv"], np.nan)
        mgm["int24"] = np.where(mgm["float_mv"] > 0, mgm["d24"] / mgm["float_mv"], np.nan)
        for w, col in [(6, "int6"), (24, "int24")]:
            recs = []
            for (ym, f), g in mgm.groupby(["ym", "factor"], observed=True):
                top = g[g["grp"] == N_GROUPS - 1]
                bot = g[g["grp"] == 0]
                if len(top) < 10 or len(bot) < 10:
                    continue
                wt = top["float_mv"] / top["float_mv"].sum()
                wb = bot["float_mv"] / bot["float_mv"].sum()
                recs.append({"ym": ym, "factor": f,
                             "crowd_w": float((top[col].fillna(0) * wt).sum() - (bot[col].fillna(0) * wb).sum())})
            cw = pd.DataFrame(recs)
            cw["ym"] = pd.PeriodIndex(cw["ym"], freq="M")
            m3 = fr.merge(cw, on=["ym", "factor"], how="inner").sort_values(["factor", "date"])
            m3["fwd1"] = m3["ls_fwd1"]
            m3["z"] = (m3["crowd_w"] - m3["crowd_w"].mean()) / m3["crowd_w"].std()
            d = m3.dropna(subset=["z", "fwd1"])
            if len(d) >= 40:
                beta, t, p, pp, n = fit(d, "z")
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

    t5 = pd.DataFrame(rows)
    t5.to_csv(REPORTS / "table5_robustness.csv", index=False, encoding="utf-8-sig")
    log.info("\n★ Table 5 稳健性：\n" + t5.round(4).to_string())
    heartbeat("11 稳健性结束", log)


if __name__ == "__main__":
    main()
