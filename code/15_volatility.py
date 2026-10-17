"""
15_volatility.py —— comomentum 是否预测因子的**实现波动与偏度**

这个检验的目的是把两种完全不同的解释分开：

  Lou & Polk (2022) 在美股记录了三件事——comomentum 预测
  ①更低的多空收益、②更高的波动、③更负的偏度。
  本文发现①在 A 股不成立（comomentum 对收益毫无预测力，安慰剂 p 0.35~0.97）。
  但这有两种解释：

    （甲）**测度没坏**，只是不预测收益。那么它应当**仍能预测波动与偏度**——
          毕竟它度量的是残差相关，而残差相关高本来就意味着共同因子暴露大。
    （乙）**我们的实现有问题**（因子分组、FF3 构造、相关矩阵哪一步错了）。
          那么它连波动也预测不了。

  这两种解释对论文的含义完全不同：（甲）下结论是「间接测度在 A 股捕捉不到
  可交易的拥挤」；（乙）下结论就只能是「本文的 comomentum 实现失败」——
  后者会让整篇文章站不住。所以必须做这个检验并如实报出。

做法：用 06 写出的 `factor_daily.parquet`（日频多空收益），
  对每个形成月 t，算持有期内的日频实现波动与偏度，再对拥挤度做回归。
  推断口径与主检验一致：循环平移安慰剂 p，含因子固定效应。

产出：reports/table7_volatility.csv  +  reports/volatility_check.md
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/金融科研1014/code")
from common import (CLEAN, REPORTS, get_logger, hac_ols, heartbeat, now_local,  # noqa: E402
                    placebo_pvalue, read_parquet)

log = get_logger("15_vol", "15_volatility.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
HORIZONS = [1, 3, 6]
MIN_DAYS = 8          # 一个月至少要有这么多交易日才计算实现矩


def build_realized() -> pd.DataFrame:
    """由日频多空收益算出每个 (形成月, 因子) 在**持有期**内的实现波动与偏度。"""
    d = read_parquet(CLEAN / "factor_daily.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d["ym"] = pd.PeriodIndex(d["ym"], freq="M")
    d = d.sort_values(["factor", "date"])

    # 持有月 = 形成月 + 1。先把日频收益按「持有月」聚合出月度实现矩，
    # 再按窗口长度滚动累加（波动用平方和开方，偏度用三阶矩合成）。
    value_col = "ls_fwd1" if "ls_fwd1" in d.columns else "ls_ret"
    if value_col == "ls_ret":
        log.warning("factor_daily.parquet 使用旧字段 ls_ret，按兼容路径读取；建议先重跑 06_build_factors.py")
    g = d.groupby(["factor", "ym"], observed=True)[value_col]
    mon = pd.DataFrame({
        "n_days": g.count(),
        "sum_r": g.sum(),
        "sum_r2": g.apply(lambda s: float((s ** 2).sum())),
        "sum_r3": g.apply(lambda s: float((s ** 3).sum())),
    }).reset_index()
    mon = mon[mon["n_days"] >= MIN_DAYS].copy()

    out = []
    for f, gg in mon.groupby("factor", observed=True):
        gg = gg.sort_values("ym").reset_index(drop=True)
        for h in HORIZONS:
            n = gg["n_days"].rolling(h, min_periods=h).sum()
            s1 = gg["sum_r"].rolling(h, min_periods=h).sum()
            s2 = gg["sum_r2"].rolling(h, min_periods=h).sum()
            s3 = gg["sum_r3"].rolling(h, min_periods=h).sum()
            mean = s1 / n
            var = (s2 / n) - mean ** 2
            m3 = (s3 / n) - 3 * mean * (s2 / n) + 2 * mean ** 3
            vol = np.sqrt(var.clip(lower=0))
            skew = np.where(vol > 0, m3 / (vol ** 3), np.nan)
            tmp = pd.DataFrame({
                "factor": f,
                # 形成月 = 持有窗口起点 − 1
                "ym": gg["ym"] - h,
                "horizon": h,
                "realized_vol": vol.to_numpy(),
                "realized_skew": skew,
                "n_days": n.to_numpy(),
            })
            out.append(tmp)
    r = pd.concat(out, ignore_index=True)
    r["ym"] = pd.PeriodIndex(r["ym"], freq="M")
    return r.dropna(subset=["realized_vol"])


def main() -> None:
    r = build_realized()
    cr = read_parquet(CLEAN / "crowding.parquet")
    cr["ym"] = pd.PeriodIndex(cr["ym"], freq="M")
    df = r.merge(cr, on=["ym", "factor"], how="inner")
    log.info(f"实现矩面板 {len(df)} 行 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    rows = []
    for measure in ["comomentum", "crowd_margin"]:
        if measure not in df.columns:
            continue
        for h in HORIZONS:
            for target, tname in (("realized_vol", "波动"), ("realized_skew", "偏度")):
                d = df[(df["horizon"] == h)].dropna(subset=[measure, target]).copy()
                if len(d) < 60:
                    continue
                fe = pd.get_dummies(d["factor"], prefix="FE", drop_first=True).astype(float)
                d = pd.concat([d, fe], axis=1)
                d[measure + "_z"] = (d[measure] - d[measure].mean()) / d[measure].std()
                cols = [measure + "_z"] + list(fe.columns)
                try:
                    res = hac_ols(d[target].to_numpy(), d[cols].to_numpy(), cols)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"回归失败 {measure} h={h} {target}: {e}")
                    continue
                pp, _ = placebo_pvalue(d, cols, measure + "_z", target,
                                       res[measure + "_z"]["t"], z_by="factor")
                rows.append({"measure": measure, "horizon": h, "target": tname,
                             "beta": res[measure + "_z"]["coef"],
                             "t": res[measure + "_z"]["t"],
                             "p_hac": res[measure + "_z"]["p"],
                             "p_placebo": pp, "n": res["_n"]})

    t = pd.DataFrame(rows)
    if t.empty:
        log.error("没有任何可用结果")
        return
    t.to_csv(REPORTS / "table7_volatility.csv", index=False, encoding="utf-8-sig")
    log.info("\n★ Table 7 拥挤度对实现波动/偏度的预测：\n" + t.round(4).to_string())

    # ---------- 判定 ----------
    cm = t[(t["measure"] == "comomentum") & (t["target"] == "波动")]
    mm = t[(t["measure"] == "crowd_margin") & (t["target"] == "波动")]
    md = ["# comomentum 是否预测实现波动？", "",
          f"生成时间：{now_local():%Y-%m-%d %H:%M:%S}（北京时间）", "",
          "**这个检验要回答的问题**：comomentum 在 A 股不预测收益（本文主结果），",
          "是因为【测度没坏、只是不预测收益】，还是因为【我们的实现有问题】？",
          "Lou & Polk 在美股记录到 comomentum 同时预测波动与偏度。",
          "若 comomentum 在 A 股**仍能预测波动**，就支持前者；若连波动也预测不了，",
          "则必须认真怀疑是实现的问题。", "",
          "| 测度 | 持有期 | 目标 | 系数 | HAC t | 安慰剂 p | 样本 |",
          "|---|---|---|---|---|---|---|"]
    for _, x in t.iterrows():
        md.append(f"| {x['measure']} | {x['horizon']} | {x['target']} | {x['beta']:+.4f} | "
                  f"{x['t']:+.2f} | {x['p_placebo']:.3f} | {x['n']} |")
    md += ["", "## 判定", ""]
    if len(cm) and (cm["p_placebo"] < 0.10).any():
        md.append("**comomentum 对波动有方向性预测证据，但证据不再支持强的全面复现结论。**")
        md.append("至少一个持有期的有限置换 p < 0.10；其余波动与偏度单元需按探索性结果报告。")
        md.append("这支持测度包含有信息的残差共变动，但不能证明它识别了账户级拥挤或真实持仓。")
    else:
        md.append("**comomentum 对波动也没有预测力。**")
        md.append("这**不支持**上面的解释（甲），必须认真对待解释（乙）：")
        md.append("应当回头检查因子分组、FF3 构造与相关矩阵的实现，"
                  "在排查清楚之前，不能把「间接测度在 A 股失效」当作结论写进论文。")
    if len(mm):
        md.append("")
        md.append(f"（对照：两融拥挤度对波动的安慰剂 p 为 "
                  f"{', '.join(f'{v:.3f}' for v in mm['p_placebo'])}）")
    (REPORTS / "volatility_check.md").write_text("\n".join(md), encoding="utf-8")
    log.info("已写出 reports/volatility_check.md")
    heartbeat("15 波动检验结束", log)


if __name__ == "__main__":
    main()
