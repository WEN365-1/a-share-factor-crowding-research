"""
10_tail_risk.py —— 尾部检验（Table 4）

"拥挤导致崩溃"比"拥挤导致收益略低"是个更有意思的主张，所以除了均值还要看尾部：
  1. 高 / 低拥挤期，因子收益的均值、波动、偏度、峰度与"极大回撤"频率
  2. 分位数回归：拥挤度对因子收益分布下端（10%、25%）的影响是否强于对中位数

推断口径与主检验一致：**一律报循环平移的安慰剂 p，并以其为准**。
  原因同 09：形成期 12 个月却按月重叠采样，空数据上 HAC/渐近 t 的经验 95 分位远高于 1.96。
  分位数回归的渐近 t 在 n≈59 的因子内样本里尤其不可靠，所以分位数回归改为
  **合并面板 + 因子固定效应**（n≈350），并对每个分位数单独做安慰剂校准。

产出：reports/table4_tail.csv  +  reports/table4_quantile.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "/home/user/金融科研1014/code")
from common import (CLEAN, REPORTS, get_logger, heartbeat, now_local, read_parquet,
                    placebo_pvalue, hac_ols)  # noqa: E402

log = get_logger("10_tail", "10_tail_risk.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
MIN_SHIFT_N = 30          # 因子内时序长度低于此则不参与安慰剂


def quanteq_coef(d: pd.DataFrame, cols: list[str], measure: str, y: str, q: float):
    """分位数回归，返回 (系数, 渐近 t, 是否收敛)。设计矩阵含因子固定效应。

    必须检查收敛性：默认迭代上限（1000）下，带固定效应虚拟变量的分位数回归会
    频繁触顶。用没收敛的系数去构建安慰剂零分布是不可靠的，所以不收敛的一律剔除。
    """
    X = sm.add_constant(d[cols].to_numpy(dtype=float), has_constant="add")
    r = sm.QuantReg(d[y].to_numpy(dtype=float), X).fit(q=q, max_iter=20000)
    conv = bool(getattr(r, "converged", True))
    return float(r.params[1]), float(r.tvalues[1]), conv


def quanteq_placebo(d: pd.DataFrame, cols: list[str], measure: str, y: str,
                    q: float, obs_t: float) -> tuple[float, int, int]:
    """分位数回归的 pooled circular-shift permutation p 值。

    每次平移后仍在完整 pooled panel 上估计相同的固定效应分位数回归，
    不再把各因子单独估计得到的 t 值混成观测 pooled statistic 的零分布。
    """
    frame = d.sort_values(["factor", "ym"]).copy()
    groups = [g for _, g in frame.groupby("factor", observed=True, sort=False)]
    if not groups or any(len(g) < MIN_SHIFT_N for g in groups):
        return float("nan"), 0, 0
    n_shifts = max(len(g) for g in groups) - 1
    ts, n_bad = [], 0
    for k in range(1, n_shifts + 1):
        perm = frame.copy()
        for _, idx in frame.groupby("factor", observed=True, sort=False).groups.items():
            ordered = frame.loc[list(idx)].sort_values("ym").index
            values = frame.loc[ordered, measure].to_numpy(dtype=float)
            perm.loc[ordered, measure] = np.roll(values, k % len(values))
        try:
            _, t, conv = quanteq_coef(perm, cols, measure, y, q)
        except Exception:  # noqa: BLE001
            n_bad += 1
            continue
        if not conv or not np.isfinite(t):
            n_bad += 1
            continue
        ts.append(abs(t))
    if not ts:
        return float("nan"), 0, n_bad
    return float((1.0 + np.sum(np.asarray(ts) >= abs(obs_t))) / (1.0 + len(ts))), len(ts), n_bad


def main() -> None:
    fr = read_parquet(CLEAN / "factor_returns.parquet")
    fr["date"] = pd.to_datetime(fr["date"])
    fr["ym"] = fr["date"].dt.to_period("M")
    cr = read_parquet(CLEAN / "crowding.parquet")
    cr["ym"] = pd.PeriodIndex(cr["ym"], freq="M")
    df = fr.merge(cr, on=["ym", "factor"], how="inner").sort_values(["factor", "date"])
    log.info(f"尾部检验面板 {len(df)} 行 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    # 每个因子的历史 5% 分位，作为"极大回撤"的阈值
    q5 = fr.groupby("factor", observed=True)["ls_fwd1"].quantile(0.05).to_dict()

    # ---------- 1. 描述性对比（高/低拥挤期）----------
    # 高/低按**因子内**中位数划分：拥挤度在不同因子间的水平本就不同，
    # 用全样本中位数切会把"哪个因子整体更拥挤"混进来。
    rows = []
    for measure in ["comomentum", "crowd_margin"]:
        if measure not in df.columns:
            continue
        for f in FACTORS:
            s = df[df["factor"] == f].dropna(subset=[measure, "ls_fwd1"])
            if len(s) < 24:
                continue
            med = s[measure].median()
            hi = s[s[measure] >= med]["ls_fwd1"]
            lo = s[s[measure] < med]["ls_fwd1"]
            if len(hi) < 8 or len(lo) < 8:
                continue
            thr = q5.get(f, np.nan)
            rows.append({
                "measure": measure, "factor": f, "n": len(s),
                "高拥挤_均值": hi.mean(), "低拥挤_均值": lo.mean(),
                "高拥挤_偏度": hi.skew(), "低拥挤_偏度": lo.skew(),
                "高拥挤_峰度": hi.kurt(), "低拥挤_峰度": lo.kurt(),
                "高拥挤_标准差": hi.std(), "低拥挤_标准差": lo.std(),
                "高拥挤_崩溃频率": (hi <= thr).mean(),
                "低拥挤_崩溃频率": (lo <= thr).mean(),
            })
    t4 = pd.DataFrame(rows)
    t4.to_csv(REPORTS / "table4_tail.csv", index=False, encoding="utf-8-sig")
    log.info("\n★ Table 4 尾部检验：\n" + t4.round(4).to_string())

    # ---------- 2. 分位数回归（合并面板 + 因子固定效应）----------
    # 原实现是逐因子单独跑（n≈59），分位数回归在这么小的样本里系数与渐近 t 都不可靠。
    # 改为合并面板加因子固定效应：因子内的水平差被吸收，n 提到 ~350。
    qrows = []
    for measure in ["comomentum", "crowd_margin"]:
        if measure not in df.columns:
            continue
        d = df.dropna(subset=[measure, "ls_fwd1"]).copy()
        if len(d) < 120:
            continue
        fe = pd.get_dummies(d["factor"], prefix="FE", drop_first=True).astype(float)
        d = pd.concat([d, fe], axis=1)
        z = (d[measure] - d[measure].mean()) / d[measure].std()
        d[measure + "_z"] = z
        cols = [measure + "_z"] + [c for c in d.columns if c.startswith("FE_")]
        for q in QUANTILES:
            try:
                coef, tt, conv = quanteq_coef(d, cols, measure + "_z", "ls_fwd1", q)
            except Exception as e:  # noqa: BLE001
                log.warning(f"分位数回归失败 {measure} q={q}: {e}")
                continue
            if not conv:
                log.warning(f"分位数回归未收敛（已跳过）{measure} q={q}")
                continue
            pp, ndraw, n_bad = quanteq_placebo(d, cols, measure + "_z", "ls_fwd1", q, tt)
            qrows.append({"measure": measure, "quantile": q, "coef": coef,
                          "t": tt, "p_placebo": pp, "n_placebo": ndraw,
                          "n_placebo_no_converge": n_bad, "n": len(d)})
    tq = pd.DataFrame(qrows)
    tq.to_csv(REPORTS / "table4_quantile.csv", index=False, encoding="utf-8-sig")
    if len(tq):
        piv = tq.pivot_table(index="measure", columns="quantile", values="coef")
        pivp = tq.pivot_table(index="measure", columns="quantile", values="p_placebo")
        log.info("\n★ 分位数回归系数（合并面板，因子固定效应）：\n" + piv.round(4).to_string())
        log.info("\n★ 对应安慰剂 p 值：\n" + pivp.round(3).to_string())
        log.info("读法：若拥挤度对左尾（q=0.10/0.25）的系数明显强于对中位数（q=0.50），"
                 "说明它影响的是尾部而不只是均值——这正是文献里被称为"
                 "「拥挤预测尾部风险而非平均收益」的那类发现。")
    heartbeat("10 尾部检验结束", log)


if __name__ == "__main__":
    main()
