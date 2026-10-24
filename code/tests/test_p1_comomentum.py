from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
os.environ["FIN_ROOT"] = str(PROJECT)


def load_module():
    path = PROJECT / "code" / "19_comomentum_faithful.py"
    spec = importlib.util.spec_from_file_location("comomentum_19", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FaithfulComomentumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def test_weekly_compounding_and_missing_return(self):
        daily = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
            "code": ["a", "a", "b", "b"],
            "ret": [0.10, 0.10, 0.05, np.nan],
            "close": [10.0, 10.1, 20.0, 20.1],
            "float_mv": [100.0] * 4,
            "industry": ["x"] * 4,
        })
        w = self.m.aggregate_weekly(daily).set_index("code")
        self.assertAlmostEqual(w.loc["a", "ret_week"], 0.21)
        self.assertTrue(np.isnan(w.loc["b", "ret_week"]))
        self.assertEqual(w.loc["a", "week_end"], pd.Timestamp("2024-01-05"))

    def test_leave_one_out_matches_naive(self):
        rng = np.random.default_rng(7)
        Res = pd.DataFrame(rng.normal(size=(8, 4)), columns=list("abcd"))
        out = self.m.leave_one_out_correlations(
            Res, list("abcd"), min_stock_weeks=8, min_pairs_weeks=8, min_other_stocks=2,
            min_per_leg=4)
        manual = []
        for code in Res.columns:
            others = Res[[c for c in Res.columns if c != code]].mean(axis=1)
            manual.append(Res[code].corr(others))
        self.assertEqual(out["n"], 4)
        self.assertAlmostEqual(out["comomentum"], float(np.mean(manual)), places=12)
        self.assertAlmostEqual(out["mean_pair_weeks"], 8.0)

    def test_pairwise_matches_direct_correlation(self):
        rng = np.random.default_rng(11)
        Res = pd.DataFrame(rng.normal(size=(8, 4)), columns=list("abcd"))
        out = self.m.pairwise_correlations(
            Res, list("abcd"), min_stock_weeks=8, min_pairs_weeks=8, min_per_leg=4)
        C = Res.corr().to_numpy()
        expected = C[np.triu_indices_from(C, k=1)].mean()
        self.assertAlmostEqual(out["comomentum"], expected, places=12)
        self.assertAlmostEqual(out["mean_pair_weeks"], 8.0)

    def test_formation_window_excludes_future_week(self):
        idx = pd.DatetimeIndex(pd.to_datetime(
            ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"]))
        lo, hi = self.m.formation_window_indices(idx, pd.Timestamp("2024-01-15"), window=2)
        self.assertEqual((lo, hi), (0, 2))
        self.assertEqual(list(idx[lo:hi]), list(idx[:2]))

    def test_local_microstructure_screen(self):
        g = pd.DataFrame({"float_mv": np.arange(1.0, 11.0),
                          "close": np.repeat(5.0, 10)})
        g.loc[2, "close"] = 4.99
        keep = self.m.microstructure_screen(g)
        self.assertFalse(bool(keep.iloc[0]))  # 最小市值十分位
        self.assertFalse(bool(keep.iloc[2]))  # 低于 5 元
        self.assertTrue(bool(keep.iloc[1]))
        self.assertTrue(bool(keep.iloc[9]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
