"""Terminal-return audit tests; synthetic data only."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
os.environ["FIN_ROOT"] = str(PROJECT)


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


terminal = load_module("28_terminal_return_audit")


class TerminalReturnTests(unittest.TestCase):
    def test_raw_month_keeps_st_days_and_drops_nontrading_days(self):
        daily = pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "pctChg": [10.0, 10.0, 999.0],
            "tradestatus": [1, 1, 0],
            "isST": [0, 1, 1],
        })
        out = terminal.raw_monthly_returns(daily)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "raw_n_days"], 2)
        self.assertEqual(out.loc[0, "raw_st_days"], 1)
        self.assertAlmostEqual(out.loc[0, "raw_mret"], 0.21)

    def test_missing_retained_day_invalidates_month(self):
        daily = pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03"],
            "pctChg": [10.0, None], "tradestatus": [1, 1],
        })
        out = terminal.raw_monthly_returns(daily)
        self.assertTrue(pd.isna(out.loc[0, "raw_mret"]))
        self.assertEqual(out.loc[0, "raw_n_ret"], 1)

    def test_only_outdate_inside_holding_month_is_terminal(self):
        frame = pd.DataFrame({
            "ym": pd.period_range("2020-01", periods=3, freq="M"),
            "outDate": ["2020-02-15", "2020-02-29", "2020-01-31"],
        })
        out = terminal.add_event_labels(frame, pd.Period("2020-12", freq="M"))
        self.assertEqual(out["terminal_in_holding_month"].tolist(), [True, False, False])
        self.assertFalse(out["sample_right_censored"].any())

    def test_sample_right_censoring_is_not_a_terminal_event(self):
        frame = pd.DataFrame({"ym": [pd.Period("2020-12", freq="M")],
                              "outDate": [None]})
        out = terminal.add_event_labels(frame, pd.Period("2020-12", freq="M"))
        self.assertTrue(out.loc[0, "sample_right_censored"])
        self.assertFalse(out.loc[0, "terminal_in_holding_month"])

    def test_stress_payoffs_remain_in_weight_denominator(self):
        leg = pd.DataFrame({
            "float_mv": [60.0, 40.0], "ret_next": [np.nan, 0.10],
            "raw_ret_next": [np.nan, 0.10],
            "terminal_in_holding_month": [True, False],
        })
        out = terminal.leg_scenarios(leg)
        self.assertAlmostEqual(out["raw_realized"][0], 0.10)
        self.assertAlmostEqual(out["raw_realized"][1], 0.40)
        self.assertAlmostEqual(out["terminal_zero_stress"][0], 0.04)
        self.assertAlmostEqual(out["terminal_zero_stress"][1], 1.0)
        self.assertAlmostEqual(out["terminal_total_loss_stress"][0], -0.56)
        self.assertAlmostEqual(out["terminal_total_loss_stress"][1], 1.0)

    def test_nonterminal_gap_is_not_imputed(self):
        leg = pd.DataFrame({
            "float_mv": [60.0, 40.0], "ret_next": [np.nan, 0.10],
            "raw_ret_next": [np.nan, 0.10],
            "terminal_in_holding_month": [False, False],
            "sample_right_censored": [True, False],
        })
        out = terminal.leg_scenarios(leg)
        for name in ("raw_realized", "terminal_zero_stress", "terminal_total_loss_stress"):
            self.assertAlmostEqual(out[name][0], 0.10)
            self.assertAlmostEqual(out[name][1], 0.40)

    def test_right_censored_terminal_gap_is_not_imputed(self):
        leg = pd.DataFrame({
            "float_mv": [60.0, 40.0], "ret_next": [np.nan, 0.10],
            "raw_ret_next": [np.nan, 0.10],
            "terminal_in_holding_month": [True, False],
            "sample_right_censored": [True, False],
        })
        out = terminal.leg_scenarios(leg)
        for name in ("raw_realized", "terminal_zero_stress", "terminal_total_loss_stress"):
            self.assertAlmostEqual(out[name][0], 0.10)
            self.assertAlmostEqual(out[name][1], 0.40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
