"""23_lp_second_moments.py —— 同口径长期收益、实现矩与崩溃频率检验。

设计先于结果固定：
* 旧日频 comomentum、LP core、LP screen、融资余额使用相同样本、规格 D、
  因子固定效应、HAC(12) 和因子内 circular-shift 推断。
* 融资余额补做 12 个月复利收益及六个因子个别斜率。
* 实现矩包括日收益波动/偏度、日收益 < -1% 频率、周收益偏度、
  周收益 < -5% 频率；后两类对应 Lou--Polk (2022) Table 3D 与 IA Table A2。
* 多重检验在每个 target × horizon 的四测度族内报告 BH 与 Bonferroni；
  因子个别结果只作探索性证据。
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
                    placebo_pvalue, read_parquet, regression_frame)

log = get_logger("23_lp_second_moments", "23_lp_second_moments.log")
MEASURES = ["comomentum", "comomentum_lp", "comomentum_lp_screen", "crowd_margin"]
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]


def load_21():
    path = Path(__file__).resolve().with_name("21_comomentum_robustness.py")
    spec = importlib.util.spec_from_file_location("comomentum_21", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compound_forward(s: pd.Series, h: int) -> pd.Series:
    """ls_fwd1[t] 已是 t+1 月收益，因此窗口从当前行开始，不额外 shift(-1)。"""
    return (1.0 + s).rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-(h - 1)) - 1.0


def build_outcomes() -> pd.DataFrame:
    """从日频多空收益构造持有期实现矩，日期只由形成月 ym 与持有长度决定。"""
    d = read_parquet(CLEAN / "factor_daily.parquet").copy()
    d["date"] = pd.to_datetime(d["date"])
    d["ym"] = pd.PeriodIndex(d["ym"], freq="M")
    d = d.sort_values(["factor", "date"])
    g = d.groupby(["factor", "ym"], observed=True)["ls_fwd1"]
    mon = pd.DataFrame({
        "n_days": g.count(),
        "sum_r": g.sum(),
        "sum_r2": g.apply(lambda s: float((s ** 2).sum())),
        "sum_r3": g.apply(lambda s: float((s ** 3).sum())),
        "bad_days": g.apply(lambda s: int((s < -0.01).sum())),
    }).reset_index()
    # 短月份记为无效，而不是删除后让 rolling 跨日历缺口拼接。
    mon.loc[mon["n_days"] < 8, ["sum_r", "sum_r2", "sum_r3", "bad_days"]] = np.nan

    # 跨月周只能复利一次；按 Friday 所在月份归入持有月，再映射回形成月。
    d["week"] = d["date"].dt.to_period("W-FRI")
    wg = d.groupby(["factor", "week"], observed=True)["ls_fwd1"]
    weekly = wg.apply(lambda s: float((1.0 + s).prod() - 1.0)).rename("week_ret").reset_index()
    weekly["ym"] = weekly["week"].dt.end_time.dt.to_period("M") - 1
    wm = weekly.groupby(["factor", "ym"], observed=True)["week_ret"].agg(
        n_weeks="count", bad_weeks=lambda s: int((s < -0.05).sum()),
        w1="sum", w2=lambda s: float((s ** 2).sum()),
        w3=lambda s: float((s ** 3).sum())).reset_index()
    mon = mon.merge(wm, on=["factor", "ym"], how="left", validate="one_to_one")

    def fwd_sum(s: pd.Series, h: int) -> pd.Series:
        # 行 t 的滚动窗口先在 t+h-1 完成，再移回形成月 t。
        return s.rolling(h, min_periods=h).sum().shift(-(h - 1))

    out = []
    for factor, gg in mon.groupby("factor", observed=True):
        # 重建连续月历，任何缺月都会使覆盖该缺月的持有期结果保持缺失。
        full_months = pd.period_range(gg["ym"].min(), gg["ym"].max(), freq="M")
        gg = gg.set_index("ym").reindex(full_months).rename_axis("ym").reset_index()
        for h in [1, 3, 6, 12]:
            n = fwd_sum(gg["n_days"], h)
            s1 = fwd_sum(gg["sum_r"], h)
            s2 = fwd_sum(gg["sum_r2"], h)
            s3 = fwd_sum(gg["sum_r3"], h)
            mean = s1 / n
            var = s2 / n - mean ** 2
            m3 = s3 / n - 3 * mean * (s2 / n) + 2 * mean ** 3
            vol = np.sqrt(var.clip(lower=0))
            # 行 t 对应 t+1 ... t+h 的实现收益；daily 文件 ym=t 已是形成月。
            tmp = pd.DataFrame({"factor": factor, "ym": gg["ym"], "horizon": h,
                                "daily_vol": vol, "daily_skew": np.where(vol > 0, m3 / vol ** 3, np.nan),
                                "daily_bad_share": fwd_sum(gg["bad_days"], h) / n,
                                "n_days": n})
            if h in [6, 12]:
                nw = fwd_sum(gg["n_weeks"], h)
                # 合并各月周收益的三阶矩，而非对每月偏度取平均。
                ws1 = fwd_sum(gg["w1"], h)
                ws2 = fwd_sum(gg["w2"], h)
                ws3 = fwd_sum(gg["w3"], h)
                wmean = ws1 / nw
                wvar = ws2 / nw - wmean ** 2
                wm3 = ws3 / nw - 3 * wmean * (ws2 / nw) + 2 * wmean ** 3
                wvol = np.sqrt(wvar.clip(lower=0))
                tmp["weekly_skew"] = np.where(wvol > 0, wm3 / wvol ** 3, np.nan)
                tmp["weekly_bad_share"] = fwd_sum(gg["bad_weeks"], h) / nw
                tmp["n_weeks"] = nw
            out.append(tmp)
    return pd.concat(out, ignore_index=True)


def add_adjustments(out: pd.DataFrame, keys: list[str], family: int) -> pd.DataFrame:
    out = out.copy()
    out["p_bonferroni"] = out.groupby(keys)["p_placebo"].transform(
        lambda s: (family * s).clip(upper=1.0))
    out["p_bh"] = np.nan
    for _, idx in out.groupby(keys).groups.items():
        vals = out.loc[idx, "p_placebo"]
        if vals.notna().all():
            out.loc[idx, "p_bh"] = multipletests(vals, method="fdr_bh")[1]
    return out


def margin_returns(df: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    rows = []
    for factor in ["pooled"] + FACTORS:
        frame = df if factor == "pooled" else df[df["factor"] == factor]
        z, y = "crowd_margin_z", "fwd12_comp"
        d = regression_frame(frame.dropna(subset=[z, y] + controls))
        cols = [z] + controls
        r = hac_ols(d[y].to_numpy(), d[cols].to_numpy(), cols, lags=12)
        pp, nd = placebo_pvalue(d, cols, z, y, r[z]["t"], hac_lags=12)
        rows.append({"sample": factor, "measure": "crowd_margin", "horizon": 12,
                     "return_type": "compound", "hac_lags": 12, "beta": r[z]["coef"],
                     "se": r[z]["se"], "t": r[z]["t"], "p_hac": r[z]["p"],
                     "p_placebo": pp, "n_placebo": nd, "n": r["_n"], "r2": r["_r2"]})
    out = pd.DataFrame(rows)
    factor_rows = out["sample"] != "pooled"
    out["p_bonferroni_6"] = np.nan
    out["p_bh_6"] = np.nan
    out.loc[factor_rows, "p_bonferroni_6"] = (6 * out.loc[factor_rows, "p_placebo"]).clip(upper=1.0)
    out.loc[factor_rows, "p_bh_6"] = multipletests(out.loc[factor_rows, "p_placebo"], method="fdr_bh")[1]
    out.to_csv(REPORTS / "margin_h12_compound.csv", index=False, encoding="utf-8-sig")
    return out


def moment_regressions(df: pd.DataFrame, controls: list[str], outcomes: pd.DataFrame,
                       sample: str = "pooled") -> pd.DataFrame:
    rows = []
    for h in [1, 3, 6, 12]:
        targets = ["daily_vol", "daily_skew", "daily_bad_share"]
        if h in [6, 12]:
            targets += ["weekly_skew", "weekly_bad_share"]
        for target in targets:
            # 四测度严格共同样本，避免样本差被误读为测度差。
            required = [m + "_z" for m in MEASURES] + controls
            base = df.merge(outcomes[outcomes["horizon"] == h], on=["ym", "factor"],
                            how="inner", validate="one_to_one")
            base = regression_frame(base.dropna(subset=required + [target]))
            for measure in MEASURES:
                z = measure + "_z"
                cols = [z] + controls
                r = hac_ols(base[target].to_numpy(), base[cols].to_numpy(), cols, lags=12)
                pp, nd = placebo_pvalue(base, cols, z, target, r[z]["t"], hac_lags=12)
                rows.append({"sample": sample, "measure": measure, "horizon": h, "target": target,
                             "hac_lags": 12, "beta": r[z]["coef"], "se": r[z]["se"],
                             "t": r[z]["t"], "p_hac": r[z]["p"], "p_placebo": pp,
                             "n_placebo": nd, "n": r["_n"], "r2": r["_r2"]})
    out = add_adjustments(pd.DataFrame(rows), ["horizon", "target"], len(MEASURES))
    filename = "lp_second_moments.csv" if sample == "pooled" else f"lp_second_moments_{sample}.csv"
    out.to_csv(REPORTS / filename, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    df, controls = load_21().build_panel()
    df = regression_frame(df)
    log.info("同口径检验面板 %s 行 | %s", len(df), now_local())
    margins = margin_returns(df, controls)
    log.info("融资余额 12 个月复利结果\n%s", margins.round(5).to_string(index=False))
    outcomes = build_outcomes()
    outcomes.to_parquet(CLEAN / "holding_period_outcomes.parquet", index=False)
    moments = moment_regressions(df, controls, outcomes)
    log.info("实现矩与崩溃频率结果\n%s",
             moments[["measure", "horizon", "target", "beta", "t", "p_placebo", "p_bh", "n"]]
             .round(5).to_string(index=False))
    mom = moment_regressions(df[df["factor"] == "MOM"].copy(), controls, outcomes, sample="MOM")
    log.info("MOM 单独实现矩与崩溃频率结果\n%s",
             mom[["measure", "horizon", "target", "beta", "t", "p_placebo", "p_bh", "n"]]
             .round(5).to_string(index=False))


if __name__ == "__main__":
    main()
