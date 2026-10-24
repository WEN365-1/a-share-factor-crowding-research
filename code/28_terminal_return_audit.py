"""Audit holding-period returns without applying formation ST filters ex post.

The production panel removes ST stock-days before monthly returns are formed. ST
status is a formation-universe rule, not a reason to erase subsequent realisations.
This script independently compounds raw, unadjusted-provider daily percentage
changes on normal trading days while retaining ST days. It never overwrites
production inputs.

Only terminal events whose outDate falls inside the holding calendar month enter
the zero-return and total-loss stress scenarios. Other missing observations and
sample-end right censoring are never imputed. Stress scenarios are assumptions,
not estimates of Chinese delisting or share-exchange payoffs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, DAILY, REPORTS, formation_universe, get_logger,  # noqa: E402
                    read_parquet, unit_path)

log = get_logger("28_terminal_returns", "28_terminal_return_audit.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
N_GROUPS = 10
VARIANTS = ["production", "raw_realized", "terminal_zero_stress",
            "terminal_total_loss_stress"]


def raw_monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound normal-trading-day returns, including ST days, by calendar month."""
    required = {"date", "pctChg", "tradestatus"}
    missing = required.difference(daily.columns)
    if missing:
        raise KeyError(f"Raw daily input lacks {sorted(missing)}")
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["tradestatus"] = pd.to_numeric(d["tradestatus"], errors="coerce")
    d["ret"] = pd.to_numeric(d["pctChg"], errors="coerce") / 100.0
    d["st_day"] = (pd.to_numeric(d.get("isST", 0), errors="coerce").fillna(0).eq(1)
                   if "isST" in d.columns else False)
    d = d[d["tradestatus"].eq(1) & d["date"].notna()].copy()
    if d["date"].duplicated().any():
        raise ValueError("Duplicate normal-trading-day rows within a stock")
    d["ym"] = d["date"].dt.to_period("M")
    d["gross"] = 1.0 + d["ret"]
    out = (d.groupby("ym", observed=True)
           .agg(raw_mret=("gross", lambda s: s.prod(min_count=len(s)) - 1.0),
                raw_n_days=("date", "size"), raw_n_ret=("ret", "count"),
                raw_st_days=("st_day", "sum"))
           .reset_index())
    # min_count=len(s) makes any missing retained-day return invalidate the month.
    invalid = out["raw_n_ret"] < out["raw_n_days"]
    out.loc[invalid, "raw_mret"] = np.nan
    return out[["ym", "raw_mret", "raw_n_days", "raw_n_ret", "raw_st_days"]]


def add_event_labels(frame: pd.DataFrame, raw_sample_end: pd.Period) -> pd.DataFrame:
    """Label terminal months and right censoring without assigning any payoff."""
    out = frame.copy()
    out["ym"] = pd.PeriodIndex(out["ym"], freq="M")
    out["hold_ym"] = out["ym"] + 1
    out["outDate"] = pd.to_datetime(out["outDate"], errors="coerce")
    hold_start = out["hold_ym"].dt.start_time
    hold_end = out["hold_ym"].dt.end_time.dt.normalize()
    out["terminal_in_holding_month"] = (
        out["outDate"].notna() & out["outDate"].ge(hold_start)
        & out["outDate"].le(hold_end))
    out["sample_right_censored"] = out["hold_ym"] > raw_sample_end
    return out


def weighted_return(leg: pd.DataFrame, weights: pd.Series, value_column: str,
                    terminal_payoff: float | None = None) -> tuple[float, float, int]:
    """Realise a frozen-weight leg; optional payoff applies only to terminal gaps."""
    values = pd.to_numeric(leg[value_column], errors="coerce").copy()
    if terminal_payoff is not None:
        right_censored = (leg["sample_right_censored"].fillna(False)
                          if "sample_right_censored" in leg.columns
                          else pd.Series(False, index=leg.index))
        terminal_gap = (leg["terminal_in_holding_month"].fillna(False)
                        & values.isna() & ~right_censored)
        values.loc[terminal_gap] = terminal_payoff
    valid = values.notna()
    coverage = float(weights[valid].sum())
    if not valid.any() or coverage <= 0:
        return np.nan, coverage, 0
    ret = float((values[valid] * weights[valid] / coverage).sum())
    return ret, coverage, int(valid.sum())


def leg_scenarios(leg: pd.DataFrame) -> dict[str, tuple[float, float, int]]:
    """Use one full-leg formation-weight denominator before valid-return handling."""
    if leg.empty or leg["float_mv"].sum() <= 0:
        raise ValueError("A factor leg must have positive formation market value")
    weights = leg["float_mv"] / leg["float_mv"].sum()
    return {
        "production": weighted_return(leg, weights, "ret_next"),
        "raw_realized": weighted_return(leg, weights, "raw_ret_next"),
        "terminal_zero_stress": weighted_return(
            leg, weights, "raw_ret_next", terminal_payoff=0.0),
        "terminal_total_loss_stress": weighted_return(
            leg, weights, "raw_ret_next", terminal_payoff=-1.0),
    }


def _difference_stats(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {"n": 0}
    return {
        "n": int(len(x)), "mean": float(x.mean()),
        "mean_abs": float(x.abs().mean()),
        "max_abs": float(x.abs().max()),
        "share_abs_gt_1bp": float((x.abs() > .0001).mean()),
        "share_abs_gt_10bp": float((x.abs() > .001).mean()),
    }


def main() -> None:
    monthly = read_parquet(CLEAN / "monthly.parquet").copy()
    monthly["code"] = monthly["code"].astype(str)
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    assert not monthly.duplicated(["code", "ym"]).any()

    # Read only the stocks actually present in the formation panel; aggregate one
    # file at a time so the unfiltered raw daily panel is never held in memory.
    raw_parts, missing_files, empty_files = [], [], []
    codes = sorted(monthly["code"].unique())
    for i, code in enumerate(codes, start=1):
        path = unit_path(DAILY / "3", code)
        if not path.is_file():
            missing_files.append(code)
            continue
        daily = pd.read_parquet(path, columns=["date", "pctChg", "tradestatus", "isST"])
        if daily.empty:
            empty_files.append(code)
            continue
        part = raw_monthly_returns(daily)
        part["code"] = code
        raw_parts.append(part)
        if i % 1000 == 0:
            log.info("Raw monthly return reconstruction: %d/%d stocks", i, len(codes))
    if not raw_parts:
        raise RuntimeError("No raw daily files could be reconstructed")
    raw = pd.concat(raw_parts, ignore_index=True)
    raw_sample_end = raw["ym"].max()
    raw_next = raw[["code", "ym", "raw_mret", "raw_n_days", "raw_st_days"]].copy()
    raw_next["ym"] = raw_next["ym"] - 1
    raw_next = raw_next.rename(columns={
        "raw_mret": "raw_ret_next", "raw_n_days": "raw_next_n_days",
        "raw_st_days": "raw_next_st_days"})

    universe = read_parquet(CLEAN / "universe.parquet")
    universe["code"] = universe["code"].astype(str)
    meta = universe[["code", "code_name", "outDate", "status"]].drop_duplicates("code")
    panel = (monthly.merge(raw_next, on=["code", "ym"], how="left", validate="one_to_one")
             .merge(meta, on=["code"], how="left", validate="many_to_one"))
    panel = add_event_labels(panel, raw_sample_end)
    panel["raw_minus_production"] = panel["raw_ret_next"] - panel["ret_next"]
    close = np.isclose(panel["raw_ret_next"], panel["ret_next"],
                       rtol=1e-6, atol=1e-7, equal_nan=True)
    recovered = panel["ret_next"].isna() & panel["raw_ret_next"].notna()
    changed = (panel["ret_next"].notna() & panel["raw_ret_next"].notna() & ~close)
    raw_gap = panel["raw_ret_next"].isna()
    panel["return_audit_status"] = "unchanged"
    panel.loc[changed, "return_audit_status"] = "revised_observed_return"
    panel.loc[recovered, "return_audit_status"] = "recovered_missing_production"
    panel.loc[raw_gap & panel["sample_right_censored"], "return_audit_status"] = "sample_right_censored"
    panel.loc[raw_gap & ~panel["sample_right_censored"]
              & panel["terminal_in_holding_month"], "return_audit_status"] = "terminal_unobserved"
    panel.loc[raw_gap & ~panel["sample_right_censored"]
              & ~panel["terminal_in_holding_month"], "return_audit_status"] = "unresolved_nonterminal"

    event_mask = (~close | panel["terminal_in_holding_month"]
                  | panel["return_audit_status"].ne("unchanged"))
    event_columns = ["code", "code_name", "ym", "hold_ym", "outDate", "status",
                     "ret_next", "raw_ret_next", "raw_minus_production",
                     "raw_next_n_days", "raw_next_st_days", "terminal_in_holding_month",
                     "sample_right_censored", "return_audit_status"]
    events = panel.loc[event_mask, event_columns].sort_values(["ym", "code"])
    events.to_csv(REPORTS / "terminal_return_events.csv", index=False, encoding="utf-8-sig")

    # Rebuild the exact production extreme legs, changing only realisation returns.
    records, exposure_records, stock_exposure_frames = [], [], []
    eligible = formation_universe(panel)
    for ym, group in eligible.groupby("ym", observed=True):
        for factor in FACTORS:
            d = group.dropna(subset=[factor])
            if len(d) < N_GROUPS * 5:
                continue
            d = d.assign(grp=pd.qcut(d[factor].rank(method="first"), N_GROUPS, labels=False))
            row = {"ym": str(ym), "factor": factor}
            for side, grp in (("top", N_GROUPS - 1), ("bot", 0)):
                leg = d[d["grp"].eq(grp)]
                scenarios = leg_scenarios(leg)
                w = leg["float_mv"] / leg["float_mv"].sum()
                terminal = leg["terminal_in_holding_month"].fillna(False)
                terminal_gap = (terminal & leg["raw_ret_next"].isna()
                                & ~leg["sample_right_censored"].fillna(False))
                recovered_leg = leg["ret_next"].isna() & leg["raw_ret_next"].notna()
                exposure_records.append({
                    "ym": str(ym), "factor": factor, "side": side,
                    "terminal_formation_weight": float(w[terminal].sum()),
                    "terminal_unobserved_formation_weight": float(w[terminal_gap].sum()),
                    "recovered_formation_weight": float(w[recovered_leg].sum()),
                    "n_terminal": int(terminal.sum()),
                    "n_terminal_unobserved": int(terminal_gap.sum()),
                })
                for variant, (ret, coverage, n_ret) in scenarios.items():
                    row[f"{side}_{variant}"] = ret
                    row[f"{side}_{variant}_coverage"] = coverage
                    row[f"{side}_{variant}_n_ret"] = n_ret
            for variant in VARIANTS:
                row[variant] = row[f"top_{variant}"] - row[f"bot_{variant}"]
            records.append(row)
            # Denominators are computed before selecting only the interesting rows.
            extreme = d[d["grp"].isin([0, N_GROUPS - 1])].copy()
            extreme["factor"] = factor
            extreme["side"] = np.where(extreme["grp"].eq(N_GROUPS - 1), "top", "bot")
            extreme["leg_mv"] = extreme.groupby("side")["float_mv"].transform("sum")
            interesting = (extreme["terminal_in_holding_month"].fillna(False)
                           | (extreme["ret_next"].isna() & extreme["raw_ret_next"].notna()))
            if interesting.any():
                stock_exp = extreme.loc[interesting, [
                    "code", "code_name", "ym", "factor", "side", "float_mv", "leg_mv",
                    "outDate", "ret_next", "raw_ret_next", "terminal_in_holding_month"]].copy()
                stock_exp["formation_weight_within_leg"] = stock_exp["float_mv"] / stock_exp["leg_mv"]
                stock_exposure_frames.append(stock_exp.drop(columns="leg_mv"))

    factor_returns = pd.DataFrame(records)
    factor_returns.to_csv(REPORTS / "terminal_factor_returns.csv", index=False, encoding="utf-8-sig")

    aggregate_exposure = pd.DataFrame(exposure_records)
    aggregate_exposure.to_csv(REPORTS / "terminal_leg_exposure.csv", index=False, encoding="utf-8-sig")
    stock_exposure = (pd.concat(stock_exposure_frames, ignore_index=True)
                      if stock_exposure_frames else
                      pd.DataFrame(columns=["code", "ym", "factor", "side"]))
    stock_exposure.to_csv(REPORTS / "terminal_stock_factor_exposure.csv",
                          index=False, encoding="utf-8-sig")

    production = read_parquet(CLEAN / "factor_returns.parquet")
    production["ym"] = production["ym"].astype(str)
    check = production[["ym", "factor", "ls_fwd1"]].merge(
        factor_returns[["ym", "factor", "production"]], on=["ym", "factor"],
        validate="one_to_one")
    baseline_error = (check["production"] - check["ls_fwd1"]).abs()
    assert float(baseline_error.max()) < 1e-6, "Production leg reconstruction changed"

    compare = production[["ym", "factor", "ls_fwd1"]].merge(
        factor_returns[["ym", "factor"] + VARIANTS[1:]], on=["ym", "factor"], validate="one_to_one")
    per_factor = []
    for factor in FACTORS:
        f = compare[compare["factor"].eq(factor)]
        item = {"factor": factor}
        for variant in VARIANTS[1:]:
            item[variant] = _difference_stats(f[variant] - f["ls_fwd1"])
        per_factor.append(item)

    terminal = panel["terminal_in_holding_month"]
    summary = {
        "status": "PASS",
        "scope": "Independent raw daily realisation audit and explicitly labelled terminal-payoff stress scenarios",
        "st_filter_interpretation": "ST is a formation-universe filter; subsequent ST trading-day returns remain realisations",
        "raw_sample_last_month": str(raw_sample_end),
        "monthly_rows": int(len(panel)), "stocks": int(len(codes)),
        "missing_raw_files": missing_files, "empty_raw_files": empty_files,
        "production_ret_next_missing": int(panel["ret_next"].isna().sum()),
        "raw_ret_next_missing": int(panel["raw_ret_next"].isna().sum()),
        "recovered_observed_returns": int(recovered.sum()),
        "revised_observed_returns": int(changed.sum()),
        "materially_revised_over_1bp": int((changed & panel["raw_minus_production"].abs().gt(.0001)).sum()),
        "revised_with_holding_period_st_days": int((changed & panel["raw_next_st_days"].gt(0)).sum()),
        "revised_without_holding_period_st_days": int((changed & ~panel["raw_next_st_days"].gt(0)).sum()),
        "production_observed_but_raw_missing": int((panel["ret_next"].notna() & raw_gap).sum()),
        "revised_return_difference_stats": _difference_stats(panel.loc[changed, "raw_minus_production"]),
        "status_counts": {str(k): int(v) for k, v in
                          panel["return_audit_status"].value_counts().items()},
        "terminal_holding_month_events": int(terminal.sum()),
        "terminal_observed_raw_returns": int((terminal & panel["raw_ret_next"].notna()).sum()),
        "terminal_unobserved_raw_returns": int((terminal & panel["raw_ret_next"].isna()).sum()),
        "terminal_unobserved_noncensored": int((terminal & panel["raw_ret_next"].isna()
                                                & ~panel["sample_right_censored"]).sum()),
        "terminal_events_with_sample_right_censoring": int(
            (terminal & panel["sample_right_censored"]).sum()),
        "production_reconstruction_max_abs_error": float(baseline_error.max()),
        "per_factor_factor_return_differences": per_factor,
        "leg_exposure": {
            "max_terminal_unobserved_formation_weight": float(
                aggregate_exposure["terminal_unobserved_formation_weight"].max()),
            "max_recovered_formation_weight": float(
                aggregate_exposure["recovered_formation_weight"].max()),
            "legs_with_terminal_unobserved": int(
                aggregate_exposure["terminal_unobserved_formation_weight"].gt(0).sum()),
            "legs_with_recovered_returns": int(
                aggregate_exposure["recovered_formation_weight"].gt(0).sum()),
        },
        "scenario_rules": {
            "raw_realized": "Observable raw returns only; renormalise over nonmissing formation weight",
            "terminal_zero_stress": "Set 0% only for unobserved returns with outDate in the holding month",
            "terminal_total_loss_stress": "Set -100% under the same terminal-event rule; conservative stress, not a payoff estimate",
            "excluded_from_imputation": "Nonterminal gaps and sample-end right censoring",
            "weights": "Scenario payoffs remain in the valid-weight denominator",
        },
    }
    (REPORTS / "terminal_return_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("Terminal return audit\n%s", json.dumps(summary, ensure_ascii=False, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
