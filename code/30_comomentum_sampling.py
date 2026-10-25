"""30_comomentum_sampling.py —— comomentum 股票对抽样稳定性审计。

预先固定 20 个种子（含生产种子），每个种子独立重复同一无放回抽样。
每个月只估一次 FF3 残差，所有种子复用该残差矩阵。随后对每个种子分别
执行规格 D 与因子内循环平移推断；种子不是新增时间序列观测，不参与
扩大回归 N，也不按系数或显著性筛选。

产出：逐种子测度、跨种子月度分布、逐种子回归/相关及机器可读摘要。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN, REPORTS, get_logger, read_parquet, write_parquet  # noqa: E402


def load_module(stem: str):
    path = Path(__file__).resolve().with_name(stem + ".py")
    spec = importlib.util.spec_from_file_location(stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


crowd = load_module("08_crowding")
reg = load_module("09_regressions")
log = get_logger("30_comomentum_sampling", "30_comomentum_sampling.log")

# 在看到任何结果前固定：生产种子 + 其后连续 19 个整数，不做结果导向筛选。
SEEDS = tuple(range(crowd.SEED, crowd.SEED + 20))
DRAW_PATH = CLEAN / "comomentum_seed_draws.parquet"


def build_membership(monthly: pd.DataFrame) -> pd.DataFrame:
    """逐项复用 08 的形成月分组规则，不引入新的股票池定义。"""
    from common import formation_universe
    rows = []
    for ym, g in monthly.groupby("ym", observed=True):
        g = formation_universe(g)
        for f in crowd.FACTORS:
            d = g.dropna(subset=[f])
            if len(d) < crowd.N_GROUPS * 5:
                continue
            try:
                d = d.assign(grp=pd.qcut(d[f].rank(method="first"), crowd.N_GROUPS, labels=False))
            except ValueError:
                continue
            rows.append(d[["ym", "code", "float_mv"]].assign(factor=f, grp=d["grp"].values))
    return pd.concat(rows, ignore_index=True)


def construct_draws() -> pd.DataFrame:
    monthly = read_parquet(CLEAN / "monthly.parquet")
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    monthly["code"] = monthly["code"].astype(str)
    membership = build_membership(monthly)
    daily = crowd.load_daily()
    ff3 = crowd.build_daily_ff3(daily, membership)
    R, X = crowd.build_return_matrix(daily, ff3)
    del daily, ff3

    cal = read_parquet(CLEAN / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    cal["ym"] = cal["date"].dt.to_period("M")
    month_ends = sorted(cal.groupby("ym", observed=True)["date"].max().tolist())
    mem_by = {ym: {f: gg for f, gg in g.groupby("factor", observed=True)}
              for ym, g in membership.groupby("ym", observed=True)}
    rngs = {seed: np.random.default_rng(seed) for seed in SEEDS}
    dates = R.index.to_numpy()
    rows = []
    for t in month_ends:
        hi = int(np.searchsorted(dates, np.datetime64(t), side="right"))
        lo = hi - crowd.FORM_WINDOW
        if lo < 0 or hi - lo < crowd.MIN_WIN_DAYS:
            continue
        ym = pd.Period(t, freq="M")
        legs = mem_by.get(ym)
        if not legs:
            continue
        Res = crowd.window_residuals(R, X, lo, hi)
        # 对每个种子，因子与腿的抽样顺序与 08 完全一致；生产种子因此逐值复现。
        for seed in SEEDS:
            for f in crowd.FACTORS:
                sub = legs.get(f)
                if sub is None or sub.empty:
                    continue
                vals, details = [], []
                for grp in (crowd.N_GROUPS - 1, 0):
                    codes = sub[sub["grp"] == grp]["code"].tolist()
                    leg = crowd.sampled_leg_correlation(Res, codes, rngs[seed])
                    vals.append(leg["correlation"])
                    details.append(leg)
                if all(np.isfinite(vals)):
                    rows.append({"ym": str(ym), "factor": f, "seed": seed,
                                 "comomentum": float(np.mean(vals)),
                                 "n_eligible_long": details[0]["n_eligible"],
                                 "n_eligible_short": details[1]["n_eligible"],
                                 "n_sampled_long": details[0]["n_sampled"],
                                 "n_sampled_short": details[1]["n_sampled"],
                                 "n_pairs_long": details[0]["n_pairs"],
                                 "n_pairs_short": details[1]["n_pairs"]})
        del Res
        log.info("多种子测度完成 %s，累计 %s 行", ym, len(rows))
    out = pd.DataFrame(rows)
    write_parquet(out, DRAW_PATH)
    return out


def summarize_draws(draws: pd.DataFrame) -> pd.DataFrame:
    """按因子—月总结测度抽样分布，不把种子重复堆成回归观测。"""
    def one(g: pd.DataFrame) -> pd.Series:
        x = g["comomentum"].dropna()
        pos = float((x > 0).mean()) if len(x) else np.nan
        neg = float((x < 0).mean()) if len(x) else np.nan
        return pd.Series({"n_draws": len(x), "mean": x.mean(), "sd": x.std(ddof=1),
                          "min": x.min(), "q05": x.quantile(.05), "q25": x.quantile(.25),
                          "median": x.median(), "q75": x.quantile(.75), "q95": x.quantile(.95),
                          "max": x.max(), "positive_fraction": pos, "negative_fraction": neg,
                          "sign_stability": max(pos, neg) if len(x) else np.nan})
    return draws.groupby(["ym", "factor"], observed=True).apply(one, include_groups=False).reset_index()


def replace_seed_measure(base: pd.DataFrame, draw: pd.DataFrame) -> pd.DataFrame:
    """替换测度并在该种子完整回归面板上重新标准化，保持其他输入不变。"""
    out = base.drop(columns=["comomentum", "comomentum_z"], errors="ignore").merge(
        draw[["ym", "factor", "comomentum"]], on=["ym", "factor"], how="left", validate="one_to_one")
    if out["comomentum"].isna().all():
        raise ValueError("种子测度未匹配到回归面板")
    out["comomentum_z"] = ((out["comomentum"] - out["comomentum"].mean())
                           / out["comomentum"].std(ddof=1))
    return out


def run_inference(draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = reg.build_panel()
    base["ym"] = pd.PeriodIndex(base["ym"], freq="M")
    regression_rows, correlation_rows = [], []
    for seed in SEEDS:
        draw = draws[draws["seed"] == seed].copy()
        draw["ym"] = pd.PeriodIndex(draw["ym"], freq="M")
        df = replace_seed_measure(base, draw)
        cols = ["comomentum_z"] + reg.controls_for(df, "D")
        for h in reg.HORIZONS:
            d = df.dropna(subset=cols + [f"fwd{h}"]).copy()
            r, pp, ndraw, lags = reg.infer_horizon(d, cols, "comomentum_z", f"fwd{h}", h)
            regression_rows.append({"seed": seed, "measure": "comomentum", "spec": "D",
                                    "horizon": h, "hac_lags": lags,
                                    "beta": r["comomentum_z"]["coef"], "se": r["comomentum_z"]["se"],
                                    "t": r["comomentum_z"]["t"], "p_hac": r["comomentum_z"]["p"],
                                    "p_placebo": pp, "n_placebo": ndraw,
                                    "n": r["_n"], "r2": r["_r2"]})
        for f, g in df.groupby("factor", observed=True):
            s = g.dropna(subset=["comomentum", "crowd_margin"])
            correlation_rows.append({"seed": seed, "factor": f, "n_months": len(s),
                                     "pearson": s["comomentum"].corr(s["crowd_margin"]),
                                     "spearman": s["comomentum"].corr(s["crowd_margin"], method="spearman")})
        pooled = df.dropna(subset=["comomentum", "crowd_margin"]).copy()
        for c in ("comomentum", "crowd_margin"):
            pooled[c] = pooled[c] - pooled.groupby("factor", observed=True)[c].transform("mean")
        correlation_rows.append({"seed": seed, "factor": "POOLED_WITHIN", "n_months": len(pooled),
                                 "pearson": pooled["comomentum"].corr(pooled["crowd_margin"]),
                                 "spearman": pooled["comomentum"].corr(pooled["crowd_margin"], method="spearman")})
        log.info("种子 %s 的规格 D 与置换推断完成", seed)
    return pd.DataFrame(regression_rows), pd.DataFrame(correlation_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    if args.draws_only and args.inference_only:
        parser.error("两个 only 选项不能同时使用")
    draws = construct_draws() if not args.inference_only else read_parquet(DRAW_PATH)
    monthly = summarize_draws(draws)
    monthly.to_csv(REPORTS / "comomentum_sampling_monthly.csv", index=False, encoding="utf-8-sig")
    if args.draws_only:
        return
    regressions, correlations = run_inference(draws)
    regressions.to_csv(REPORTS / "comomentum_sampling_regressions.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(REPORTS / "comomentum_sampling_correlations.csv", index=False, encoding="utf-8-sig")

    production = read_parquet(CLEAN / "crowding.parquet")[["ym", "factor", "comomentum"]].copy()
    production["ym"] = production["ym"].astype(str)
    baseline = draws[draws["seed"] == crowd.SEED][["ym", "factor", "comomentum"]]
    check = production.merge(baseline, on=["ym", "factor"], suffixes=("_production", "_seed"),
                             validate="one_to_one")
    max_diff = float((check["comomentum_production"] - check["comomentum_seed"]).abs().max())
    if not np.isfinite(max_diff) or max_diff > 1e-12:
        raise AssertionError(f"生产种子未逐值复现：max diff={max_diff}")
    main_rows = pd.read_csv(REPORTS / "table2_main.csv")
    original = main_rows[(main_rows["measure"] == "comomentum") & (main_rows["spec"] == "D")]
    original = original[["horizon", "beta", "p_placebo"]].sort_values("horizon").reset_index(drop=True)
    reproduced = regressions[regressions["seed"] == crowd.SEED][["horizon", "beta", "p_placebo"]]
    reproduced = reproduced.sort_values("horizon").reset_index(drop=True)
    if not np.allclose(original["beta"], reproduced["beta"], atol=1e-12, rtol=0):
        raise AssertionError("生产种子的规格 D 系数未复现")
    if not np.allclose(original["p_placebo"], reproduced["p_placebo"], atol=1e-12, rtol=0):
        raise AssertionError("生产种子的循环平移 p 值未复现")
    summary = {
        "seeds_predeclared": list(SEEDS), "n_seeds": len(SEEDS),
        "sampling": "Without replacement, at most 100 stocks per factor-leg-month",
        "inference": "Specification D separately within each seed; grouped circular shifts with the same HAC lags",
        "not_independent_observations": "Seed replicates are never pooled to increase regression N and never selected by significance",
        "production_measure_max_abs_difference": max_diff,
        "production_regression_reproduced": True,
        "monthly_cells": int(len(monthly)),
        "monthly_sd_median": float(monthly["sd"].median()),
        "monthly_sd_q95": float(monthly["sd"].quantile(.95)),
        "monthly_sd_max": float(monthly["sd"].max()),
        "minimum_sign_stability": float(monthly["sign_stability"].min()),
        "regressions": regressions.to_dict(orient="records"),
    }
    (REPORTS / "comomentum_sampling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("多种子审计完成；生产测度最大复现差异 %.3g", max_diff)


if __name__ == "__main__":
    main()
