"""Return-realisation robustness for the specification-D return regressions.

Only the factor return series changes. Crowding, its standardisation, coverage
controls, factor fixed effects, HAC lags and circular-shift inference are held at
their production values. The common-month-end variant changes membership and is
therefore a partial timing check, not a full reconstruction of every input.

The daily-leg variant compounds available frozen-weight leg contributions. It is
an estimator-sensitivity check, not proof that missing stock-days are tradable
zero returns or that terminal/delisting payoffs are complete.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, get_logger, hac_ols, placebo_pvalue, regression_frame  # noqa: E402

log = get_logger("27_formation_robustness", "27_formation_robustness.log")
HORIZONS = [1, 3, 6, 12]
MEASURES = ["comomentum_z", "crowd_margin_z"]
VARIANTS = {
    "production": (None, None),
    "daily_leg_buy_hold": (
        "formation_daily_leg_sensitivity.csv", "daily_leg_buy_hold_return"),
    "common_month_end_partial": (
        "formation_common_month_end_sensitivity.csv", "common_month_end_return"),
}


def load_09():
    path = Path(__file__).resolve().with_name("09_regressions.py")
    spec = importlib.util.spec_from_file_location("regressions_09_for_27", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def add_return_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute momentum controls and forward sums from the active return column."""
    out = frame.sort_values(["factor", "ym"]).copy()
    grouped = out.groupby("factor", observed=True)["ls_fwd1"]
    for k in (1, 6, 12):
        out[f"past{k}"] = grouped.shift(1).rolling(k, min_periods=k).sum()
    for h in HORIZONS:
        out[f"fwd{h}"] = grouped.transform(
            lambda s: s.rolling(h, min_periods=h).sum().shift(-(h - 1)))
    return regression_frame(out)


def replace_returns(base: pd.DataFrame, alternative: pd.DataFrame,
                    return_column: str) -> pd.DataFrame:
    """Replace returns one-to-one without altering any predictor or fixed effect."""
    required = {"ym", "factor", return_column}
    missing = required.difference(alternative.columns)
    if missing:
        raise KeyError(f"Alternative return input lacks {sorted(missing)}")
    alt = alternative[["ym", "factor", return_column]].copy()
    alt["ym"] = pd.PeriodIndex(alt["ym"], freq="M")
    if alt.duplicated(["ym", "factor"]).any():
        raise ValueError("Alternative returns are not unique by formation month and factor")
    protected = [c for c in base.columns if c not in
                 ["ls_fwd1", "past1", "past6", "past12"]
                 + [f"fwd{h}" for h in HORIZONS]]
    out = base[protected].merge(alt, on=["ym", "factor"], how="left", validate="one_to_one")
    out = out.rename(columns={return_column: "ls_fwd1"})
    return add_return_windows(out)


def build_panels() -> tuple[dict[str, pd.DataFrame], list[str]]:
    module09 = load_09()
    production = add_return_windows(module09.build_panel())
    controls = module09.controls_for(production, "D")
    panels = {"production": production}
    for name, (filename, column) in VARIANTS.items():
        if filename is not None:
            panels[name] = replace_returns(
                production, pd.read_csv(REPORTS / filename), column)
    return panels, controls


def estimate(frame: pd.DataFrame, controls: list[str], measure: str,
             horizon: int, variant: str) -> dict:
    y = f"fwd{horizon}"
    cols = [measure] + controls
    d = regression_frame(frame.dropna(subset=cols + [y]))
    if len(d) < 40:
        raise ValueError(f"Effective sample too small: {variant} {measure} h={horizon}")
    lags = 12 if horizon == 12 else 6
    result = hac_ols(d[y].to_numpy(), d[cols].to_numpy(), cols, lags=lags)
    pp, ndraw = placebo_pvalue(
        d, cols, measure, y, result[measure]["t"], hac_lags=lags)
    return {
        "variant": variant, "measure": measure.replace("_z", ""),
        "spec": "D", "horizon": horizon, "hac_lags": lags,
        "beta": result[measure]["coef"], "se": result[measure]["se"],
        "t": result[measure]["t"], "p_hac": result[measure]["p"],
        "p_placebo": pp, "n_placebo": ndraw, "n": result["_n"],
        "r2": result["_r2"],
    }


def main() -> None:
    panels, controls = build_panels()
    # Momentum controls must vary with the return estimator; every other predictor
    # and control is required to remain byte-for-byte equal at the data-frame level.
    invariant_columns = ["comomentum_z", "crowd_margin_z"] + [
        c for c in controls if c not in {"past1", "past6", "past12"}]
    for name, frame in panels.items():
        pd.testing.assert_frame_equal(
            frame[invariant_columns].reset_index(drop=True),
            panels["production"][invariant_columns].reset_index(drop=True),
            check_dtype=False, check_exact=True)

    rows = [estimate(frame, controls, measure, h, name)
            for name, frame in panels.items()
            for measure in MEASURES for h in HORIZONS]
    out = pd.DataFrame(rows)
    baseline = out[out["variant"] == "production"].set_index(["measure", "horizon"])
    out["beta_minus_production"] = out.apply(
        lambda r: r["beta"] - baseline.loc[(r["measure"], r["horizon"]), "beta"], axis=1)
    out.to_csv(REPORTS / "formation_robustness_regressions.csv",
               index=False, encoding="utf-8-sig")

    reference_path = REPORTS / "table2_main.csv"
    reference_comparison = {"checked": False}
    if reference_path.exists():
        ref = pd.read_csv(reference_path)
        ref = ref[ref["spec"] == "D"].set_index(["measure", "horizon"])
        common_idx = baseline.index.intersection(ref.index)
        max_error = float(np.max(np.abs(
            baseline.loc[common_idx, "beta"].to_numpy()
            - ref.loc[common_idx, "beta"].to_numpy())))
        assert max_error < 1e-12, "Production specification-D reconstruction changed"
        reference_comparison = {"checked": True, "rows": int(len(common_idx)),
                                "max_abs_beta_error": max_error}

    core = out[(out["measure"] == "crowd_margin") & (out["horizon"] == 6)]
    summary = {
        "status": "PASS",
        "scope": "Specification-D return-realisation sensitivity; not a model-selection exercise",
        "rows": int(len(out)),
        "reference_comparison": reference_comparison,
        "predictors_and_controls_identical_across_variants": True,
        "multiple_testing": "No additional significance claim or selection across variants; all four horizons reported",
        "daily_leg_caveat": "Available-day compounding does not establish tradable zero returns or complete terminal/delisting payoffs",
        "common_month_end_caveat": "Alternative membership and returns only; crowding and coverage controls remain production values",
        "core_margin_h6": core[[
            "variant", "beta", "t", "p_placebo", "n", "beta_minus_production"
        ]].to_dict("records"),
    }
    (REPORTS / "formation_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Formation robustness\n%s",
             out[["variant", "measure", "horizon", "beta", "t", "p_placebo", "n"]]
             .round(6).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
