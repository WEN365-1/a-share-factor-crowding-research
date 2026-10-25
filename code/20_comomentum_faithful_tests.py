"""
20_comomentum_faithful_tests.py —— LP comomentum 对照回归

三类证据分开报告：
  1. pooled_own：各测度自己的有效样本；
  2. pooled_common：旧日频、LP core、LP screen 三者共同有效样本；
  3. MOM_only：仅在动量因子上检验，对应 Lou–Polk 的原始研究对象。

全部使用规格 D 的控制变量与同一 pooled circular-shift 推断；h=12 使用
HAC(12)，其余期限使用 HAC(6)，观测与置换的滞后阶数一致。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, REPORTS, get_logger, hac_ols, now_local,  # noqa: E402
                    placebo_pvalue, read_parquet, regression_frame, time_index)

log = get_logger("20_comomentum_faithful_tests", "20_comomentum_faithful_tests.log")
MEASURES = ["comomentum", "comomentum_lp", "comomentum_lp_screen"]
HORIZONS = [1, 3, 6, 12]


def load_09():
    path = Path(__file__).resolve().with_name("09_regressions.py")
    spec = importlib.util.spec_from_file_location("regressions_09", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_panel():
    module09 = load_09()
    df = module09.build_panel()
    new = read_parquet(CLEAN / "comomentum_faithful.parquet").copy()
    new["ym"] = pd.PeriodIndex(new["ym"], freq="M")
    df = df.merge(new, on=["ym", "factor"], how="left", validate="one_to_one")
    for c in ["comomentum_lp", "comomentum_lp_screen"]:
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
    controls = module09.controls_for(df, "D")
    return regression_frame(df), controls


def estimate(df: pd.DataFrame, controls: list[str], measure: str, horizon: int,
             sample: str, common_measures: list[str] | None = None,
             use_time: bool = True) -> dict:
    z = measure + "_z"
    required = [z] + controls + [f"fwd{horizon}"]
    if common_measures:
        required += [m + "_z" for m in common_measures]
    d = regression_frame(df.dropna(subset=list(dict.fromkeys(required))))
    if len(d) < 40:
        raise ValueError(f"有效样本过小：{sample} {measure} h={horizon}, N={len(d)}")
    cols = [z] + controls
    lags = 12 if horizon == 12 else 6
    # pooled 样本是因子 × 月堆叠面板（A 类）；MOM_only 每个 ym 只有一行（B 类）。
    tvals = time_index(d["ym"]) if use_time else None
    r = hac_ols(d[f"fwd{horizon}"].to_numpy(), d[cols].to_numpy(), cols, lags=lags,
                time=tvals)
    pp, ndraw = placebo_pvalue(d, cols, z, f"fwd{horizon}", r[z]["t"], hac_lags=lags,
                               use_time=use_time)
    return {"sample": sample, "measure": measure, "horizon": horizon, "hac_lags": lags,
            "beta": r[z]["coef"], "se": r[z]["se"], "t": r[z]["t"],
            "p_hac": r[z]["p"], "p_placebo": pp, "n_placebo": ndraw,
            "n": r["_n"], "r2": r["_r2"]}


def main() -> None:
    df, controls = build_panel()
    log.info("LP 对照回归面板 %s 行 | %s", len(df), now_local())
    rows = []
    for h in HORIZONS:
        for measure in MEASURES:
            rows.append(estimate(df, controls, measure, h, "pooled_own"))
        for measure in MEASURES:
            rows.append(estimate(df, controls, measure, h, "pooled_common", MEASURES))
        mom = df[df["factor"] == "MOM"].copy()
        for measure in MEASURES:
            rows.append(estimate(mom, controls, measure, h, "MOM_only", use_time=False))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "comomentum_faithful_regressions.csv", index=False, encoding="utf-8-sig")
    log.info("LP 对照回归结果\n%s",
             out[["sample", "measure", "horizon", "beta", "t", "p_placebo", "n"]]
             .round(5).to_string(index=False))


if __name__ == "__main__":
    main()
