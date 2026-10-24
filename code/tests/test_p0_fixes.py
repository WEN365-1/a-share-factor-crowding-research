"""P0 修复的轻量回归测试；只使用临时合成数据，不触碰原始数据。"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


common = load_module("common")
flow = load_module("18_flow_proxy")
crowding = load_module("08_crowding")
regressions = load_module("09_regressions")
extensions = load_module("16_extension_analyses")


class P0FixTests(unittest.TestCase):
    def test_short_month_and_future_return_not_used(self):
        frame = pd.DataFrame({
            "code": ["keep_short_month", "drop_too_suspended", "keep_missing_forward"],
            "float_mv": [100.0, 100.0, 100.0],
            "n_days": [5, 4, 5],
            "calendar_days": [15, 15, 15],
            "isST": [0, 0, 0],
            "ret_next": [0.01, 0.01, np.nan],
        })
        kept = common.formation_universe(frame)
        self.assertEqual(set(kept["code"]), {"keep_short_month", "keep_missing_forward"})

    def test_formation_st_filter_does_not_require_deleting_history(self):
        frame = pd.DataFrame({
            "code": ["tradable", "st_at_formation", "unknown_st"],
            "float_mv": [100.0, 100.0, 100.0],
            "n_days": [20, 20, 20], "calendar_days": [20, 20, 20],
            "isST": [0, 1, np.nan],
        })
        kept = common.formation_universe(frame)
        self.assertEqual(kept["code"].tolist(), ["tradable"])

    def test_missing_forward_return_reweights_frozen_weights(self):
        leg = pd.DataFrame({"code": ["a", "b", "c"], "ret_next": [0.10, np.nan, -0.02]})
        weights = pd.Series([0.50, 0.25, 0.25])
        ret, coverage, n = common.realize_leg_return(leg, weights)
        self.assertAlmostEqual(ret, 0.10 * (0.50 / 0.75) - 0.02 * (0.25 / 0.75))
        self.assertAlmostEqual(coverage, 0.75)
        self.assertEqual(n, 2)

    def test_szse_without_repayment_remains_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "szse.parquet"
            pd.DataFrame({
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["sz.000001", "sz.000001"],
                "fin_buy": [100.0, 200.0],
                "exchange": ["SZSE", "SZSE"],
            }).to_parquet(fp, index=False)
            daily = flow.read_margin_flow_file(fp)
            daily["ym"] = pd.PeriodIndex(["2024-01", "2024-01"], freq="M")
            monthly = flow.aggregate_flow_monthly(daily)
            self.assertAlmostEqual(monthly.loc[0, "fin_buy_sum"], 300.0)
            self.assertTrue(pd.isna(monthly.loc[0, "fin_repay_sum"]))
            self.assertTrue(pd.isna(monthly.loc[0, "fin_net_sum"]))

    def test_stale_snapshot_invalidates_balance_and_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            margin_dir = Path(tmp) / "margin"
            margin_dir.mkdir()
            pd.DataFrame({
                "date": pd.to_datetime(["2024-01-02", "2024-03-01"]),
                "code": ["sh.600000", "sh.600000"],
                "margin_balance": [100.0, 130.0],
            }).to_parquet(margin_dir / "sample.parquet", index=False)
            old_margin = crowding.MARGIN
            crowding.MARGIN = margin_dir
            try:
                dates = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-04-30"))
                out = crowding.load_margin_monthly(dates).set_index("ym")
            finally:
                crowding.MARGIN = old_margin
            feb = pd.Period("2024-02", freq="M")
            mar = pd.Period("2024-03", freq="M")
            self.assertGreater(out.loc[feb, "stale_days"], 45)
            self.assertTrue(bool(out.loc[feb, "stale_excluded"]))
            self.assertTrue(pd.isna(out.loc[feb, "margin_balance"]))
            self.assertTrue(pd.isna(out.loc[mar, "d_mb"]))
            self.assertTrue(bool(out.loc[mar, "has_history"]))

    def test_observed_and_permuted_design_have_identical_order(self):
        idx = pd.MultiIndex.from_product([
            pd.period_range("2020-01", periods=31, freq="M"), ["A", "B"]
        ], names=["ym", "factor"]).to_frame(index=False)
        idx["m"] = np.arange(len(idx), dtype=float)
        idx["fe"] = (idx["factor"] == "B").astype(float)
        idx["y"] = np.linspace(-1, 1, len(idx))
        shuffled = idx.sample(frac=1, random_state=7)
        observed = common.regression_frame(shuffled)
        permuted_frames = []
        observed_y = []
        original_shift = common._grouped_circular_shift

        def capture_shift(frame, measure, z_by, shift):
            perm = original_shift(frame, measure, z_by, shift)
            permuted_frames.append(perm)
            return perm

        def fake_hac(y, X, names, lags=6):
            observed_y.append(np.asarray(y))
            return {"m": {"t": 1.0}}

        with patch.object(common, "_grouped_circular_shift", capture_shift), \
             patch.object(common, "hac_ols", fake_hac):
            null = common.placebo_null_dist(shuffled, ["m", "fe"], "m", "y")
        self.assertEqual(len(null), 30)
        for perm, y in zip(permuted_frames, observed_y):
            self.assertTrue(perm.index.equals(observed.index))
            np.testing.assert_array_equal(y, perm["y"].to_numpy())

    def test_spec_d_controls_both_leg_coverages(self):
        frame = pd.DataFrame(columns=[
            "FE_B", "past1", "past6", "past12", "level_margin_z",
            "cover_long_z", "cover_short_z",
        ])
        for module in (regressions, extensions):
            controls = module.controls_for(frame, "D")
            self.assertIn("cover_long_z", controls)
            self.assertIn("cover_short_z", controls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
