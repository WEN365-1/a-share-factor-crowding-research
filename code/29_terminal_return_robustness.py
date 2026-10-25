"""Regression consequences of raw realisations and terminal-payoff stress tests.

All crowding predictors, standardisation, fixed effects, balance/coverage controls,
HAC lags and circular-shift inference remain at production values. Only realised
factor returns and return-derived momentum controls change. Every measure and all
four horizons are reported; variants are not selected by statistical significance.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, get_logger  # noqa: E402

log = get_logger("29_terminal_robustness", "29_terminal_return_robustness.log")
HORIZONS = [1, 3, 6, 12]
MEASURES = ["comomentum_z", "crowd_margin_z"]
VARIANTS = ["production", "raw_realized", "terminal_zero_stress",
            "terminal_total_loss_stress"]


def load_27():
    path = Path(__file__).resolve().with_name("27_formation_robustness.py")
    spec = importlib.util.spec_from_file_location("formation_27_for_29", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_panels() -> tuple[dict[str, pd.DataFrame], list[str]]:
    module27 = load_27()
    production, controls = module27.build_panels()
    base = production["production"]
    alternatives = pd.read_csv(REPORTS / "terminal_factor_returns.csv")
    panels = {"production": base}
    for name in VARIANTS[1:]:
        panels[name] = module27.replace_returns(base, alternatives, name)
    return panels, controls


def main() -> None:
    module27 = load_27()
    panels, controls = build_panels()
    invariant_columns = MEASURES + [
        c for c in controls if c not in {"past1", "past6", "past12"}]
    for frame in panels.values():
        pd.testing.assert_frame_equal(
            frame[invariant_columns].reset_index(drop=True),
            panels["production"][invariant_columns].reset_index(drop=True),
            check_dtype=False, check_exact=True)

    rows = [module27.estimate(frame, controls, measure, horizon, variant)
            for variant, frame in panels.items()
            for measure in MEASURES for horizon in HORIZONS]
    out = pd.DataFrame(rows)
    baseline = out[out["variant"].eq("production")].set_index(["measure", "horizon"])
    out["beta_minus_production"] = out.apply(
        lambda r: r["beta"] - baseline.loc[(r["measure"], r["horizon"]), "beta"], axis=1)
    out.to_csv(REPORTS / "terminal_return_regressions.csv", index=False, encoding="utf-8-sig")

    reference = pd.read_csv(REPORTS / "table2_main.csv")
    reference = reference[reference["spec"].eq("D")].set_index(["measure", "horizon"])
    common_idx = baseline.index.intersection(reference.index)
    max_error = float(np.max(np.abs(
        baseline.loc[common_idx, "beta"].to_numpy()
        - reference.loc[common_idx, "beta"].to_numpy())))
    assert max_error < 1e-12, "Production specification-D reconstruction changed"

    core = out[out["measure"].eq("crowd_margin") & out["horizon"].eq(6)]
    summary = {
        "status": "PASS",
        "scope": "Specification-D terminal-return sensitivity; stress scenarios are not payoff estimates",
        "rows": int(len(out)),
        "production_reference_max_abs_beta_error": max_error,
        "predictors_and_nonreturn_controls_identical": True,
        "multiple_testing": "All measures and four horizons reported for every variant; no selection by significance",
        "core_margin_h6": core[[
            "variant", "beta", "se", "t", "p_hac", "p_placebo", "n",
            "beta_minus_production"]].to_dict("records"),
        "interpretation_rule": "Raw observed realisations address ST filtering; terminal scenarios only bound unobserved in-month terminal payoffs",
    }
    (REPORTS / "terminal_return_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Terminal-return regressions\n%s",
             out[["variant", "measure", "horizon", "beta", "t", "p_placebo", "n"]]
             .round(6).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
