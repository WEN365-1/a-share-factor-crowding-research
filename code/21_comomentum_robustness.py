"""
21_comomentum_robustness.py —— 12 个月 comomentum 证据的对称稳健性分解

对旧日频与六种周频构造全部报告： pooled 与 MOM_only × 月收益和与复利买入
持有收益 × HAC(6) 与 HAC(12)。观测回归和循环平移零分布使用相同 HAC 滞后。
多重检验校正在每个 sample × return × HAC 的七测度族内给出，不替代原始 p。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, REPORTS, get_logger, hac_ols, now_local,  # noqa: E402
                    placebo_pvalue, read_parquet, regression_frame, time_index)

log = get_logger("21_comomentum_robustness", "21_comomentum_robustness.log")
MEASURES = ["comomentum", "comomentum_lp", "comomentum_lp_no_ind", "comomentum_lp_pair",
            "comomentum_lp_full52", "comomentum_lp_screen", "comomentum_lp_screen_full52"]


def load_20():
    path = Path(__file__).resolve().with_name("20_comomentum_faithful_tests.py")
    spec = importlib.util.spec_from_file_location("comomentum_20", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_panel():
    df, controls = load_20().build_panel()
    df = df.sort_values(["factor", "ym"]).copy()
    for c in MEASURES:
        if c + "_z" not in df:
            df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
    # 与 fwd12 完全相同的信息区间，仅把月收益和改成复利买入持有收益。
    df["fwd12_comp"] = (df.groupby("factor", observed=True)["ls_fwd1"]
                        .transform(lambda s: (1.0 + s).rolling(12, min_periods=12)
                                   .apply(np.prod, raw=True).shift(-11) - 1.0))
    return regression_frame(df), controls


def estimate(df: pd.DataFrame, controls: list[str], measure: str, sample: str,
             return_type: str, hac_lags: int, use_time: bool = True) -> dict:
    z = measure + "_z"
    y = "fwd12" if return_type == "sum" else "fwd12_comp"
    d = regression_frame(df.dropna(subset=[z, y] + controls))
    if len(d) < 40:
        raise ValueError(f"样本过小：{sample} {measure} {return_type} N={len(d)}")
    cols = [z] + controls
    # pooled（A 类）用 DK；MOM_only / 单因子切片（B 类）用普通 NW。
    tvals = time_index(d["ym"]) if use_time else None
    r = hac_ols(d[y].to_numpy(), d[cols].to_numpy(), cols, lags=hac_lags, time=tvals)
    pp, ndraw = placebo_pvalue(d, cols, z, y, r[z]["t"], hac_lags=hac_lags,
                               use_time=use_time)
    return {"sample": sample, "measure": measure, "horizon": 12,
            "return_type": return_type, "hac_lags": hac_lags,
            "beta": r[z]["coef"], "se": r[z]["se"], "t": r[z]["t"],
            "p_hac": r[z]["p"], "p_placebo": pp, "n_placebo": ndraw,
            "n": r["_n"], "r2": r["_r2"]}


def main() -> None:
    df, controls = build_panel()
    log.info("12 个月稳健性面板 %s 行 | %s", len(df), now_local())
    rows = []
    for sample, frame in [("pooled", df), ("MOM_only", df[df["factor"] == "MOM"])]:
        for return_type in ["sum", "compound"]:
            for hac_lags in [6, 12]:
                for measure in MEASURES:
                    rows.append(estimate(frame, controls, measure, sample, return_type, hac_lags,
                                         use_time=(sample == "pooled")))
    out = pd.DataFrame(rows)
    keys = ["sample", "return_type", "hac_lags"]
    out["p_bonferroni_7"] = out.groupby(keys)["p_placebo"].transform(lambda s: (7 * s).clip(upper=1.0))
    out["p_bh_7"] = np.nan
    for _, idx in out.groupby(keys).groups.items():
        out.loc[idx, "p_bh_7"] = multipletests(out.loc[idx, "p_placebo"], method="fdr_bh")[1]
    out.to_csv(REPORTS / "comomentum_h12_robustness.csv", index=False, encoding="utf-8-sig")
    log.info("12 个月稳健性结果\n%s",
             out[["sample", "measure", "return_type", "hac_lags", "beta", "t",
                  "p_placebo", "p_bonferroni_7", "n"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
