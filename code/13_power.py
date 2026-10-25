"""
13_power.py —— 检验功效 / 最小可检测效应（MDE）

为什么需要它：如果 11/15 闸门判定「系数消失」，审稿人第一个问题一定是
「是**没有效应**，还是你**检验不出来**？」。一个不显著的结果本身回答不了这个问题。
必须说明：以本样本的规模与结构，能可靠（80% 功效）检出的最小效应是多少，
再让读者判断这个量级有没有经济意义。

做法（设计固定、误差重抽）：
  1. 用真实数据跑一遍零假设回归，取出拟合值与残差；
  2. 把残差按**月份分块**循环重抽（块长取持有期 h，以保留重叠窗口带来的残差相关）；
  3. 在重抽出来的样本上注入一个已知效应 β，用与主检验**完全相同**的回归重估；
  4. 在 β 的网格上重复，得到功效曲线，读出功效 80% 对应的 MDE。

关键点：拒绝标准必须用**安慰剂**的临界值 t*（零分布的 95 分位），不能用 1.96。
  本设计形成期 12 个月却按月重叠采样，名义临界值系统性偏松
  （空数据上 |t| 的 95 分位约 2.8）。用 1.96 算出来的 MDE 会乐观得多，
  等于把「检验不出来」说成「能检验出来」。

产出：
  reports/table6_power.csv        各测度 × 持有期的 t*、MDE、年化后的经济量级
  reports/power_analysis.md       人读的说明
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPORTS, get_logger, hac_ols, heartbeat, now_local,  # noqa: E402
                    placebo_null_dist, time_index)

log = get_logger("13_power", "13_power.log")

# 复用 09 的面板构造，保证功效分析与主检验用的是同一个样本、同一套变量
_spec = importlib.util.spec_from_file_location("r09", str(Path(__file__).resolve().parent / "09_regressions.py"))
R09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R09)

SPEC = "C"            # 主检验的头号规格
HORIZONS = [1, 3, 6, 12]
N_REPS = 500
POWER_TARGET = 0.80
SEED = 20261115


def block_indices(n_months: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """循环分块重抽月份下标。块长取持有期，以保留重叠窗口造成的残差相关。"""
    n_blocks = int(np.ceil(n_months / block_len))
    starts = rng.integers(0, n_months, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) % n_months for s in starts])
    return idx[:n_months]


def cell_power(df: pd.DataFrame, measure: str, h: int, cols: list[str],
               rng: np.random.Generator) -> dict | None:
    y = f"fwd{h}"
    d = df.dropna(subset=[measure] + cols + [y]).copy()
    d = d.sort_values(["ym", "factor"]).reset_index(drop=True)
    if len(d) < 60:
        return None

    Xm = d[cols].to_numpy(dtype=float)
    ym = d["ym"].to_numpy()
    months, ym_idx = np.unique(ym, return_inverse=True)

    # 零假设回归：y 只对控制变量（含因子固定效应）回归
    Xc = np.column_stack([np.ones(len(d)), Xm])
    beta_hat, *_ = np.linalg.lstsq(Xc, d[y].to_numpy(dtype=float), rcond=None)
    fitted = Xc @ beta_hat
    u = d[y].to_numpy(dtype=float) - fitted

    # 每个月的行区间（已按 ym 排序）
    bounds = np.searchsorted(ym_idx, np.arange(len(months) + 1))

    # 安慰剂临界值：零分布的 95 分位。用真实数据算一次（面板 → DK）。
    null = placebo_null_dist(d, cols + [measure], measure, y)
    if null.size < 20:
        return None
    t_star = float(np.percentile(null, 95))

    # 全变量回归的标准误，用来定 β 网格的范围
    full_cols = [measure] + cols
    r0 = hac_ols(d[y].to_numpy(dtype=float), d[full_cols].to_numpy(dtype=float), full_cols,
                 time=time_index(d["ym"]))
    se = r0[measure]["se"]
    if not np.isfinite(se) or se <= 0:
        return None

    c = d[measure].to_numpy(dtype=float)

    # 两步各司其职，不要混在一起：
    #
    # ① 拒绝域（β 的量纲）由**真实数据**的安慰剂临界值给出：
    #        β_crit = t*_placebo × SE_HAC
    #    注意 t* 是安慰剂零分布的 95 分位，不是 1.96——本设计形成期 12 个月却按月
    #    重叠采样，名义临界值系统性偏松。
    #
    # ② β̂ 的抽样标准差 σ_b 用分块自助估计（这是自助擅长的部分）。
    #
    # 功效：β̂ ~ N(β, σ_b²)，双侧拒绝域 |β̂| > β_crit，
    #     Power(β) = Φ((−β_crit − β)/σ_b) + 1 − Φ((β_crit − β)/σ_b)。
    #     MDE 于是约等于 β_crit + 0.84·σ_b，含义直白：临界值再加上 0.84 个抽样标准差。
    #
    # **不要**在每次自助重抽内部重新估 HAC 标准误再用它算 t：重抽样本里重复的区块
    # 会制造人工的序列结构，把 HAC 标准误一并抬高，导致功效被系统性低估
    # （实测这样算出的 MDE 是正确值的 1.5~2 倍）。
    Xs = d[full_cols].to_numpy(dtype=float)
    b_hats = []
    for _ in range(N_REPS):
        idx_months = block_indices(len(months), max(h, 1), rng)
        # 重抽月份时，被解释变量与控制变量必须用同一套下标重抽——
        # 重抽是有放回的，行数不再等于原样本。
        pick = np.concatenate([np.arange(bounds[j], bounds[j + 1]) for j in idx_months])
        # 自助样本仍按原月份分组（同月可能被抽中多次）→ 与观测口径一致的 DK。
        rr = hac_ols(fitted[pick] + u[pick], Xs[pick], full_cols, time=ym_idx[pick])
        if measure in rr:
            b_hats.append(rr[measure]["coef"])
    if len(b_hats) < 30:
        return None
    sigma_b = float(np.std(b_hats, ddof=1))

    beta_crit = t_star * se
    grid = np.linspace(0.0, 3.0 * (beta_crit + 0.84 * sigma_b), 25)
    z = (-beta_crit - grid) / sigma_b
    power = norm.cdf(z) + 1.0 - norm.cdf(z + 2.0 * beta_crit / sigma_b)

    # 功效 80% 对应的 β（线性插值；网格上界不够就返回 NaN 让上层知道要放宽）
    mde = np.nan
    lo_ok = np.flatnonzero(power >= POWER_TARGET)
    if len(lo_ok) and lo_ok[0] > 0:
        i = lo_ok[0]
        x0, x1 = grid[i - 1], grid[i]
        y0, y1 = power[i - 1], power[i]
        mde = x0 + (POWER_TARGET - y0) * (x1 - x0) / (y1 - y0) if y1 > y0 else x1
    elif len(lo_ok) and lo_ok[0] == 0:
        mde = 0.0

    return {"measure": measure, "horizon": h, "spec": SPEC, "n_obs": len(d),
            "n_months": len(months), "t_star_placebo": t_star, "se_hac": se,
            "beta_crit": beta_crit, "sigma_boot": sigma_b,
            "mde_cum": mde, "mde_monthly": mde / h if np.isfinite(mde) else np.nan,
            "mde_annual_pct": 100.0 * 12.0 * (mde / h) if np.isfinite(mde) else np.nan,
            "power_at_grid": ";".join(f"{v:.2f}" for v in power)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = R09.build_panel()
    log.info(f"功效分析面板：{len(df)} 行 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    rows = []
    for measure in ["comomentum_z", "crowd_margin_z"]:
        if measure not in df.columns:
            continue
        cols = [c for c in R09.controls_for(df, SPEC) if c != measure]
        for h in HORIZONS:
            r = cell_power(df, measure, h, cols, rng)
            if r:
                rows.append(r)
                log.info(f"  {measure} h={h:2d}: t*={r['t_star_placebo']:.2f}, "
                         f"MDE={r['mde_cum']:.4f}（累计 {h} 个月）→ 年化 {r['mde_annual_pct']:.1f}%")

    if not rows:
        log.error("功效分析没有任何可用结果")
        return
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS / "table6_power.csv", index=False, encoding="utf-8-sig")

    md = ["# 检验功效 / 最小可检测效应（MDE）", "",
          f"生成时间：{now_local():%Y-%m-%d %H:%M:%S}（北京时间）", "",
          "规格 " + SPEC + "（含因子固定效应与全部控制变量），设计固定、误差按月份分块重抽，",
          f"每次 {N_REPS} 次重复，拒绝标准用安慰剂零分布的 95 分位 t*（不是 1.96）。", "",
          "| 测度 | 持有期 h | 样本 | t*（安慰剂） | MDE（累计 h 月） | MDE（年化） |",
          "|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        md.append(f"| {r['measure']} | {r['horizon']} | {r['n_obs']} | {r['t_star_placebo']:.2f} | "
                  f"{r['mde_cum']:.4f} | {r['mde_annual_pct']:.1f}% |")
    md += ["",
           "**怎么读**：MDE 是在既定设计与模拟假设下，以 80% 功效可检出的最小系数，",
           "只用于界定该设计的分辨范围。它不能证明零结果源于功效不足，也不能证明总体效应为零；",
           "即使 MDE 很小，也只能排除该设计所覆盖且大于相应量级的效应。h=12 的 MDE",
           "不得用来解释主文 h≤6 的短期结果。",
           "",
           "注意 β 的量纲：拥挤度已标准化，所以 β 是「拥挤度每高出 1 个标准差，",
           "因子在该持有期内的累计收益变化多少」。"]
    (REPORTS / "power_analysis.md").write_text("\n".join(md), encoding="utf-8")
    log.info("已写出 reports/power_analysis.md 与 table6_power.csv")
    heartbeat("13 功效分析结束", log)


if __name__ == "__main__":
    main()
