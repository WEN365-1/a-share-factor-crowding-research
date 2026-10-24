"""
15_volatility.py —— comomentum 是否预测因子的**实现波动与偏度**

这个检验考察 comomentum 与后续实现波动、偏度的关联，是不同经济原假设下的补充诊断。

  Lou & Polk (2022) 在美股记录了 comomentum 对多空收益、波动与偏度的预测关系。
  本文短期收益关联有限、周频长期结果异质；二阶矩结果可能不同，但无论显著与否，
  都不能单独证明实现正确或错误，更不能证明测度识别了账户级拥挤或真实持仓。
  实现核验仍需依靠定义对照、单元测试和独立复现，经济结果按实际证据如实报告。

做法：用 06 写出的 `factor_daily.parquet`（日频多空收益），
  对每个形成月 t，算持有期内的日频实现波动与偏度，再对拥挤度做回归。
  推断口径与主检验一致：循环平移安慰剂 p，含因子固定效应。

产出：reports/table7_volatility.csv  +  reports/volatility_check.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, REPORTS, get_logger, hac_ols, heartbeat, now_local,  # noqa: E402
                    placebo_pvalue, read_parquet)

log = get_logger("15_vol", "15_volatility.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
HORIZONS = [1, 3, 6]
MIN_DAYS = 8          # 一个月至少要有这么多交易日才计算实现矩


def build_realized() -> pd.DataFrame:
    """复用 23 的连续月历与前向窗口，避免重复实现造成月份错配。"""
    from importlib.util import module_from_spec, spec_from_file_location
    path = Path(__file__).resolve().with_name("23_lp_second_moments.py")
    spec = spec_from_file_location("second_moments_23", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    r = module.build_outcomes()
    r = r[r["horizon"].isin(HORIZONS)].copy()
    r = r.rename(columns={"daily_vol": "realized_vol", "daily_skew": "realized_skew"})
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
                d = df[(df["horizon"] == h)].dropna(subset=[measure, target]).sort_values(["ym", "factor"]).copy()
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
          "**这个检验要回答的问题**：在短期收益关联有限、周频长期结果又具有异质性的情况下，",
          "comomentum 是否与后续二阶矩存在可检测关联？Lou & Polk 在美股记录到 comomentum",
          "同时预测波动与偏度，但 A 股结果不成立本身既不能证明实现错误，成立也不能反向验证",
          "实现正确或识别了真实持仓；它只是一项具有不同经济原假设的补充诊断。", "",
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
        md.append("**这些设计未检测到 comomentum 对波动的方向性预测证据。**")
        md.append("该结果不证明实现错误，也不证明总体效应为零；应结合定义对照、单元测试、"
                  "检验分辨范围和其他独立实现诊断解释，不能据此二分推断。")
    if len(mm):
        md.append("")
        md.append(f"（对照：两融拥挤度对波动的安慰剂 p 为 "
                  f"{', '.join(f'{v:.3f}' for v in mm['p_placebo'])}）")
    (REPORTS / "volatility_check.md").write_text("\n".join(md), encoding="utf-8")
    log.info("已写出 reports/volatility_check.md")
    heartbeat("15 波动检验结束", log)


if __name__ == "__main__":
    main()
