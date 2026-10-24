"""25 — 融资快照阈值与多重校正的轻量测试；不修改原始数据。"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CODE = PROJECT / "code"
os.environ["FIN_ROOT"] = str(PROJECT)


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


crowding = load_module("08_crowding")
robust = load_module("24_margin_staleness_robustness")


class MarginStalenessTests(unittest.TestCase):
    def test_bh_adjustment_matches_reference(self):
        from statsmodels.stats.multitest import multipletests
        examples = [
            [0.04, 0.01, 0.03, 0.02],
            [0.0095238, 0.0098039, 0.0186916, 0.0208333],
            [0.2, 0.01, 0.04, 0.02],
        ]
        for values in examples:
            expected = multipletests(values, method="fdr_bh")[1]
            np.testing.assert_allclose(robust.bh_adjust(values), expected, rtol=0, atol=1e-12)

    def test_threshold_changes_only_stale_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            margin_dir = Path(tmp) / "margin"
            margin_dir.mkdir()
            pd.DataFrame({
                "date": pd.to_datetime(["2024-01-02", "2024-05-02"]),
                "code": ["sh.600000", "sh.600000"],
                "margin_balance": [100.0, 130.0],
            }).to_parquet(margin_dir / "sample.parquet", index=False)
            old_margin = crowding.MARGIN
            crowding.MARGIN = margin_dir
            try:
                dates = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-06-28"))
                out45 = crowding.load_margin_monthly(dates, max_stale_days=45).set_index("ym")
                out90 = crowding.load_margin_monthly(dates, max_stale_days=90).set_index("ym")
            finally:
                crowding.MARGIN = old_margin
            feb = pd.Period("2024-02", freq="M")
            self.assertGreater(out45.loc[feb, "stale_days"], 45)
            self.assertLess(out90.loc[feb, "stale_days"], 90)
            self.assertTrue(pd.isna(out45.loc[feb, "margin_balance"]))
            self.assertEqual(out90.loc[feb, "margin_balance"], 100.0)
            with self.assertRaises(ValueError):
                crowding.load_margin_monthly(dates, max_stale_days=-1)

    def test_stale_history_is_not_structural_zero(self):
        codes = ["c00", "c01", "c18", "c19"]
        ym = pd.Period("2024-01", freq="M")
        monthly = pd.DataFrame({"code": codes, "ym": ym, "float_mv": 1.0})
        membership = pd.DataFrame({
            "code": codes, "ym": ym, "float_mv": 1.0,
            "factor": "TEST", "grp": [9, 9, 0, 0],
        })
        mg = pd.DataFrame({
            "code": ["c00", "c01", "c19"], "ym": ["2024-01"] * 3,
            "d_mb": [10.0, np.nan, 5.0],
            "margin_balance": [110.0, np.nan, 100.0],
            "available_date": pd.to_datetime(["2024-01-03", "2023-11-01", "2024-01-03"]),
            "signal_date": pd.Timestamp("2024-01-31"),
            "stale_days": [1.0, 70.0, 1.0],
            "has_history": True,
            "stale_excluded": [False, True, False],
        })
        old_min = crowding.MIN_PER_LEG
        crowding.MIN_PER_LEG = 2
        try:
            out = crowding.build_margin_crowding(monthly, membership, mg)
        finally:
            crowding.MIN_PER_LEG = old_min
        row = out.iloc[0]
        # 多头只使用 c00 的有效变化 10；c01 是缺失而非零。
        # 空头 c18 是结构性零，c19 的变化为 5，故空头强度为 2.5。
        self.assertAlmostEqual(row["crowd_margin_long"], 10.0)
        self.assertAlmostEqual(row["crowd_margin"], 7.5)
        self.assertAlmostEqual(row["cover_long"], 0.5)
        self.assertAlmostEqual(row["cover_short"], 0.5)

    def test_baseline_comparison_detects_changed_values(self):
        frame = pd.DataFrame({
            "ym": ["2024-01"], "factor": ["MOM"],
            **{name: [1.0] for name in robust.MARGIN_COLS},
        })
        robust.assert_baseline_unchanged(frame, frame)
        changed = frame.copy()
        changed.loc[0, "crowd_margin"] = 0.0
        with self.assertRaises(AssertionError):
            robust.assert_baseline_unchanged(changed, frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
