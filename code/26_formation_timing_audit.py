"""Independent formation/realisation timing and portfolio-consistency audit.

This script never overwrites production inputs. It checks the actual monthly and
daily outputs, constructs two sensitivity series, and writes their differences:
(1) exact calendar-month-end formation and (2) daily-compounded frozen weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN, REPORTS, formation_universe, get_logger, read_parquet  # noqa: E402

log = get_logger("26_timing", "26_formation_timing_audit.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
N_GROUPS = 10


def _difference_stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "mean_abs": float(x.abs().mean()),
        "median_abs": float(x.abs().median()),
        "p95_abs": float(x.abs().quantile(.95)),
        "max_abs": float(x.abs().max()),
        "share_abs_gt_1bp": float((x.abs() > .0001).mean()),
        "share_abs_gt_10bp": float((x.abs() > .001).mean()),
        "share_abs_gt_50bp": float((x.abs() > .005).mean()),
    }


def main() -> None:
    monthly = read_parquet(CLEAN / "monthly.parquet").copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    monthly["code"] = monthly["code"].astype(str)
    cal = pd.read_parquet(CLEAN / "calendar.parquet", columns=["date"])
    cal["date"] = pd.to_datetime(cal["date"])
    cal["ym"] = cal["date"].dt.to_period("M")
    month_end = cal.groupby("ym", observed=True)["date"].max()
    monthly["month_end"] = monthly["ym"].map(month_end)
    monthly["common_month_end"] = monthly["date"].eq(monthly["month_end"])
    assert monthly["month_end"].notna().all()
    assert (monthly["date"] <= monthly["month_end"]).all()
    assert not monthly.duplicated(["code", "ym"]).any()

    # Independently reconstruct next-calendar-month returns, including months with gaps.
    expected_next = monthly[["code", "ym", "mret"]].copy()
    expected_next["ym"] = expected_next["ym"] - 1
    expected_next = expected_next.rename(columns={"mret": "expected_ret_next"})
    check = monthly[["code", "ym", "ret_next"]].merge(
        expected_next, on=["code", "ym"], how="left", validate="one_to_one")
    next_ok = np.isclose(check["ret_next"], check["expected_ret_next"],
                         rtol=1e-6, atol=1e-7, equal_nan=True)
    assert bool(next_ok.all()), "ret_next is not the next-calendar-month return"

    fr = read_parquet(CLEAN / "factor_returns.parquet").copy()
    fr["date"] = pd.to_datetime(fr["date"])
    fr["ym"] = pd.PeriodIndex(fr["ym"], freq="M")
    assert not fr.duplicated(["factor", "ym"]).any()
    assert (fr["date"].dt.to_period("M") == fr["ym"]).all()
    assert np.allclose(fr["ls_fwd1"], fr["top_ret"] - fr["bot_ret"], equal_nan=True)

    # Exact month-end membership sensitivity; all other formation rules stay unchanged.
    baseline_rows, strict_rows, membership_rows, leg_rows = [], [], [], []
    eligible = formation_universe(monthly)
    for ym, g in eligible.groupby("ym", observed=True):
        for factor in FACTORS:
            d = g.dropna(subset=[factor])
            if len(d) < N_GROUPS * 5:
                continue
            d = d.assign(grp=pd.qcut(d[factor].rank(method="first"), N_GROUPS, labels=False))
            extreme = d[d["grp"].isin([0, N_GROUPS - 1])]
            strict = d[d["common_month_end"]]
            strict_extreme = strict[strict["grp"].isin([0, N_GROUPS - 1])]
            strict_pairs = set(zip(strict_extreme["code"], strict_extreme["grp"]))
            base_pairs = set(zip(extreme["code"], extreme["grp"]))
            union = base_pairs | strict_pairs
            membership_rows.append({
                "ym": str(ym), "factor": factor,
                "n_extreme": len(extreme),
                "n_stale_extreme": int((~extreme["common_month_end"]).sum()),
                "jaccard": len(base_pairs & strict_pairs) / len(union) if union else np.nan,
            })
            for frame, target in ((extreme, baseline_rows), (strict_extreme, strict_rows)):
                for side, grp in (("top", N_GROUPS - 1), ("bot", 0)):
                    leg = frame[frame["grp"] == grp]
                    if leg.empty or leg["float_mv"].sum() <= 0:
                        continue
                    valid = leg["ret_next"].notna()
                    w = leg["float_mv"] / leg["float_mv"].sum()
                    if not valid.any() or w[valid].sum() <= 0:
                        continue
                    ret = float((leg.loc[valid, "ret_next"] * w[valid] / w[valid].sum()).sum())
                    target.append({"ym": str(ym), "factor": factor, "side": side, "ret": ret})
                    if frame is extreme:
                        leg_rows.append(leg[["code"]].assign(
                            ym=ym, factor=factor, side=side, w=w.to_numpy()))

    def spread(rows: list[dict]) -> pd.DataFrame:
        wide = pd.DataFrame(rows).pivot(index=["ym", "factor"], columns="side", values="ret")
        return (wide["top"] - wide["bot"]).rename("return").reset_index()

    production = fr[["ym", "factor", "ls_fwd1"]].copy()
    production["ym"] = production["ym"].astype(str)
    baseline = spread(baseline_rows).rename(columns={"return": "reconstructed_baseline"})
    baseline_check = production.merge(baseline, on=["ym", "factor"], validate="one_to_one")
    baseline_error = (baseline_check["reconstructed_baseline"] - baseline_check["ls_fwd1"]).abs()
    assert float(baseline_error.max()) < 1e-6, "Independent baseline reconstruction differs"
    strict = spread(strict_rows).rename(columns={"return": "common_month_end_return"})
    common = baseline.merge(strict, on=["ym", "factor"], how="outer", validate="one_to_one")
    common["difference"] = common["common_month_end_return"] - common["reconstructed_baseline"]
    common.to_csv(REPORTS / "formation_common_month_end_sensitivity.csv", index=False, encoding="utf-8-sig")

    membership = pd.DataFrame(membership_rows)
    membership.to_csv(REPORTS / "formation_membership_timing.csv", index=False, encoding="utf-8-sig")
    membership_summary = membership.groupby("factor").agg(
        months=("ym", "nunique"), mean_jaccard=("jaccard", "mean"),
        min_jaccard=("jaccard", "min"), stale_extreme=("n_stale_extreme", "sum"),
        extreme=("n_extreme", "sum")).reset_index()
    membership_summary["stale_extreme_share"] = (
        membership_summary["stale_extreme"] / membership_summary["extreme"])

    # Daily-rebalanced long-short compounding is related but not identical to
    # buying and holding each leg within the month; keep the labels explicit.
    daily = read_parquet(CLEAN / "factor_daily.parquet").copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ym"] = pd.PeriodIndex(daily["ym"], freq="M")
    assert (daily["date"].dt.to_period("M") == daily["ym"] + 1).all()
    compounded = (daily.groupby(["ym", "factor"], observed=True)["ls_fwd1"]
                  .apply(lambda s: (1.0 + s).prod() - 1.0)
                  .rename("daily_compounded_return").reset_index())
    daily_compare = production.merge(compounded.assign(ym=compounded["ym"].astype(str)),
                                     on=["ym", "factor"], how="outer", validate="one_to_one")
    daily_compare["difference"] = daily_compare["daily_compounded_return"] - daily_compare["ls_fwd1"]
    daily_compare.to_csv(REPORTS / "formation_daily_compounding_sensitivity.csv",
                         index=False, encoding="utf-8-sig")

    # Reconstruct each leg separately from frozen formation weights. This compares
    # month-level ex-post valid-return renormalisation with an implementable daily
    # calculation in which unavailable stock-days make no return contribution.
    legs = pd.concat(leg_rows, ignore_index=True)
    legs["hold_ym"] = legs["ym"] + 1
    panel_files = sorted((CLEAN / "panel").rglob("*.parquet"))
    dd = pd.concat((pd.read_parquet(f, columns=["date", "code", "ret"]) for f in panel_files),
                   ignore_index=True)
    dd["code"] = dd["code"].astype(str)
    dd["hold_ym"] = dd["date"].dt.to_period("M")
    mg = legs.merge(dd, on=["hold_ym", "code"], how="inner")
    mg["weighted_return"] = mg["w"] * mg["ret"]
    daily_side = (mg.groupby(["ym", "factor", "side", "date"], observed=True)["weighted_return"]
                  .sum().rename("daily_return").reset_index())
    monthly_side = (daily_side.groupby(["ym", "factor", "side"], observed=True)["daily_return"]
                    .apply(lambda s: (1.0 + s).prod() - 1.0).reset_index())
    side_wide = monthly_side.pivot(index=["ym", "factor"], columns="side", values="daily_return")
    side_wide["daily_leg_buy_hold_return"] = side_wide["top"] - side_wide["bot"]
    leg_compare = production.merge(side_wide[["daily_leg_buy_hold_return"]].reset_index()
                                   .assign(ym=lambda x: x["ym"].astype(str)),
                                   on=["ym", "factor"], how="outer", validate="one_to_one")
    leg_compare["difference"] = leg_compare["daily_leg_buy_hold_return"] - leg_compare["ls_fwd1"]
    leg_compare.to_csv(REPORTS / "formation_daily_leg_sensitivity.csv", index=False, encoding="utf-8-sig")

    coverage = fr[["top_ret_coverage", "bot_ret_coverage"]].stack(future_stack=True)
    per_factor = []
    for factor in FACTORS:
        c = common[common["factor"] == factor]
        d = daily_compare[daily_compare["factor"] == factor]
        b = leg_compare[leg_compare["factor"] == factor]
        per_factor.append({
            "factor": factor,
            "common_month_end": _difference_stats(c["difference"]),
            "daily_long_short_rebalanced": _difference_stats(d["difference"]),
            "daily_leg_buy_and_hold": _difference_stats(b["difference"]),
            "daily_vs_monthly_correlation": float(d[["ls_fwd1", "daily_compounded_return"]].corr().iloc[0, 1]),
            "daily_leg_vs_monthly_correlation": float(
                b[["ls_fwd1", "daily_leg_buy_hold_return"]].corr().iloc[0, 1]),
        })

    result = {
        "status": "PASS",
        "scope": "Timing and estimator-sensitivity audit; not yet a decision to replace production",
        "monthly_rows": int(len(monthly)),
        "stock_month_share_not_common_month_end": float((~monthly["common_month_end"]).mean()),
        "next_month_return_mismatches": int((~next_ok).sum()),
        "factor_return_rows": int(len(fr)),
        "baseline_reconstruction_max_abs_error": float(baseline_error.max()),
        "daily_rows": int(len(daily)),
        "daily_rows_wrong_realisation_month": int((daily["date"].dt.to_period("M") != daily["ym"] + 1).sum()),
        "formation_weight_return_coverage": {
            "mean": float(coverage.mean()), "min": float(coverage.min()),
            "share_below_99pct": float((coverage < .99).mean()),
            "share_below_95pct": float((coverage < .95).mean()),
            "share_below_90pct": float((coverage < .90).mean()),
        },
        "common_month_end_overall": _difference_stats(common["difference"]),
        "daily_long_short_rebalanced_overall": _difference_stats(daily_compare["difference"]),
        "daily_leg_buy_and_hold_overall": _difference_stats(leg_compare["difference"]),
        "per_factor": per_factor,
        "membership": membership_summary.to_dict("records"),
        "largest_common_month_end_differences": common.reindex(
            common["difference"].abs().sort_values(ascending=False).index).head(20).to_dict("records"),
        "largest_daily_compounding_differences": daily_compare.reindex(
            daily_compare["difference"].abs().sort_values(ascending=False).index).head(20).to_dict("records"),
        "largest_daily_leg_differences": leg_compare.reindex(
            leg_compare["difference"].abs().sort_values(ascending=False).index).head(20).to_dict("records"),
    }
    (REPORTS / "formation_timing_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps(result, ensure_ascii=False, default=str))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
