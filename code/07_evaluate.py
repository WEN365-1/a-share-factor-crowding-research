"""
07_evaluate.py —— 因子看板（Module 2）

对 6 个因子逐一给出：
  IC / RankIC 时序与 IC_IR、十分位分层收益与单调性、
  多空组合的年化/波动/夏普/最大回撤/换手率、IC 衰减曲线、因子相关性矩阵

产出：
  reports/factor_dashboard.csv          一行一个因子的汇总看板
  reports/ic_series.csv                 月度 IC / RankIC 时序
  reports/decile_returns.csv            十分位分层收益
  reports/ic_decay.csv                  IC 衰减曲线
  reports/factor_corr.csv               因子相关性矩阵
  reports/fig_factor_dashboard.png      因子看板图
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/金融科研1014/code")
from common import CLEAN, Gate, REPORTS, get_logger, heartbeat, now_local, read_parquet  # noqa: E402

log = get_logger("07_evaluate", "07_evaluate.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
N_GROUPS = 10


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def main() -> None:
    gate = Gate(log)
    m = read_parquet(CLEAN / "monthly.parquet")
    fr = read_parquet(CLEAN / "factor_returns.parquet")
    m["ym"] = pd.PeriodIndex(m["ym"], freq="M")
    log.info(f"月频截面 {len(m):,} 行 | {m['ym'].nunique()} 个月 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    # 与 06 保持同一截面样本。06 是先剔除「当月停牌超过 10 个交易日」的观测再分层，
    # 07 若不过滤，因子看板描述的组合就与主检验实际用的组合不是同一个，
    # 论文里「描述」与「检验」的对应关系会断掉。
    # 注意：下面算未来收益的宽表仍用**未过滤**的 m——计算个股未来 h 个月的真实收益时，
    # 少几天数据的月份也该计入复利，不该在这里丢掉。
    mc = m[m["n_days"] >= 10].copy()
    log.info(f"其中当月停牌 ≤10 个交易日的观测 {len(mc):,} 行")

    # ---------- 1. IC / RankIC ----------
    log.info("计算 IC / RankIC")
    ic_rows, dec_rows = [], []
    for ym, g in mc.groupby("ym", observed=True):
        if not gate.ok():
            log.warning("闸门触发")
            break
        g = g.dropna(subset=["ret_next"])
        if len(g) < 100:
            continue
        for f in FACTORS:
            d = g.dropna(subset=[f])
            if len(d) < 100:
                continue
            ic_rows.append({"ym": str(ym), "factor": f,
                            "ic": d[f].corr(d["ret_next"]),
                            "rank_ic": d[f].corr(d["ret_next"], method="spearman")})
            try:
                d = d.assign(grp=pd.qcut(d[f].rank(method="first"), N_GROUPS, labels=False))
            except ValueError:
                continue
            gr = d.groupby("grp", observed=True).apply(
                lambda x: np.average(x["ret_next"], weights=x["float_mv"]), include_groups=False)
            for k, v in gr.items():
                dec_rows.append({"ym": str(ym), "factor": f, "grp": int(k), "ret": float(v)})

    ic = pd.DataFrame(ic_rows)
    dec = pd.DataFrame(dec_rows)
    ic.to_csv(REPORTS / "ic_series.csv", index=False, encoding="utf-8-sig")
    dec.to_csv(REPORTS / "decile_returns.csv", index=False, encoding="utf-8-sig")

    # ---------- 2. IC 衰减曲线（对齐未来 h 月的累计收益）----------
    # 用「月 × 股票」宽表做前向滚动，避免分组 rolling 的语义陷阱：
    #   A[t]        = 从 t 起 h 个月的累计对数收益（含 t）
    #   fwd_h[t]    = A[t+1] = 从 t+1 起 h 个月的累计   ← 正是我们要的「未来 h 个月」
    log.info("计算 IC 衰减曲线")
    m["_lr"] = np.log1p(m["mret"].clip(-0.99, None))
    wide_lr = m.pivot_table(index="ym", columns="code", values="_lr", aggfunc="first")
    fut = {}
    for h in (1, 2, 3, 6, 12):
        A = wide_lr.iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]
        fut[h] = np.expm1(A.shift(-1))
    decay_rows = []
    for ym, g in mc.groupby("ym", observed=True):
        if not gate.ok():
            log.warning("闸门触发")
            break
        if ym not in fut[1].index:
            continue
        block = pd.DataFrame({f"_fut{h}": m.loc[g.index, "code"].map(fut[h].loc[ym]).values
                              for h in (1, 2, 3, 6, 12)}, index=g.index)
        g = pd.concat([g, block], axis=1)
        g = g.dropna(subset=[f"_fut{h}" for h in (1, 2, 3, 6, 12)])
        if len(g) < 100:
            continue
        for f in FACTORS:
            d = g.dropna(subset=[f])
            if len(d) < 100:
                continue
            for h in (1, 2, 3, 6, 12):
                decay_rows.append({"ym": str(ym), "factor": f, "horizon": h,
                                   "rank_ic": d[f].corr(d[f"_fut{h}"], method="spearman")})
    decay = pd.DataFrame(decay_rows)
    decay.to_csv(REPORTS / "ic_decay.csv", index=False, encoding="utf-8-sig")
    log.info(f"IC 衰减表：{len(decay)} 行")

    # ---------- 3. 因子相关性矩阵 ----------
    corr_rows = []
    for a in range(len(FACTORS)):
        for b in range(len(FACTORS)):
            xs, ys = [], []
            for ym, g in m.groupby("ym", observed=True):
                d = g.dropna(subset=[FACTORS[a], FACTORS[b]])
                if len(d) < 100:
                    continue
                xs.append(d[FACTORS[a]].rank(pct=True))
                ys.append(d[FACTORS[b]].rank(pct=True))
            if xs:
                corr_rows.append({"factor_a": FACTORS[a], "factor_b": FACTORS[b],
                                  "rank_corr": pd.concat(xs).corr(pd.concat(ys))})
    corr = pd.DataFrame(corr_rows).pivot(index="factor_a", columns="factor_b", values="rank_corr")
    corr.to_csv(REPORTS / "factor_corr.csv", encoding="utf-8-sig")
    log.info("\n因子截面秩相关矩阵：\n" + corr.round(3).to_string())

    # ---------- 4. 汇总看板 ----------
    rows = []
    for f in FACTORS:
        s = fr[fr["factor"] == f].sort_values("date")
        if s.empty:
            continue
        r = s["ls_fwd1"].values
        eq = np.cumprod(1 + r)
        ics = ic[ic["factor"] == f]
        dg = dec[dec["factor"] == f]
        mono = np.nan
        if not dg.empty:
            avg = dg.groupby("grp")["ret"].mean().sort_index()
            mono = avg.corr(pd.Series(range(len(avg)), index=avg.index), method="spearman")
        dcy = decay[decay["factor"] == f].groupby("horizon")["rank_ic"].mean()
        rows.append({
            "factor": f,
            "IC均值": ics["ic"].mean(), "IC_IR": ics["ic"].mean() / ics["ic"].std(),
            "IC_t值": ics["ic"].mean() / ics["ic"].std() * np.sqrt(len(ics)),
            "RankIC均值": ics["rank_ic"].mean(),
            "RankIC_IR": ics["rank_ic"].mean() / ics["rank_ic"].std(),
            "多空年化": (1 + r).prod() ** (12 / len(r)) - 1,
            "多空波动": r.std() * np.sqrt(12),
            "多空夏普": r.mean() / r.std() * np.sqrt(12),
            "最大回撤": max_drawdown(eq),
            "月度胜率": (r > 0).mean(),
            "分层单调性": mono,
            "IC衰减_h6": dcy.get(6, np.nan), "IC衰减_h12": dcy.get(12, np.nan),
            "样本月数": len(r),
        })
    dash = pd.DataFrame(rows).set_index("factor")
    dash.to_csv(REPORTS / "factor_dashboard.csv", encoding="utf-8-sig")
    log.info("\n" + dash.round(4).to_string())

    # ---------- 5. 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=130)
        for ax, f in zip(axes.ravel(), FACTORS):
            s = fr[fr["factor"] == f].sort_values("date")
            eq = (1 + s["ls_fwd1"]).cumprod()
            ax.plot(pd.to_datetime(s["date"]), eq, lw=1.6)
            ax.set_title(f)
            ax.grid(alpha=.3)
        fig.suptitle("Long–short cumulative NAV by factor (float-MV weighted, monthly)", y=1.01)
        fig.tight_layout()
        fig.savefig(REPORTS / "fig_factor_dashboard.png", bbox_inches="tight")
        log.info(f"图已保存 {REPORTS / 'fig_factor_dashboard.png'}")
    except Exception as e:  # noqa: BLE001
        log.warning(f"绘图失败：{e}")

    heartbeat("07 因子评估结束", log)


if __name__ == "__main__":
    main()
