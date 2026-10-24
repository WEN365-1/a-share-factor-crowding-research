"""多股票 PIT 审计的离线回归测试；不访问网络或生产数据。"""
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


pit = load_module("17_pit_audit")


def k_frame(n: int = 500) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-02", periods=n),
        "close": 10.0,
        "pbMRQ": 1.0,
        "peTTM": 10.0,
    })


def add_bps_step(k: pd.DataFrame, index: int) -> None:
    k.loc[index:, "pbMRQ"] = 10.0 / 11.0


class PitAuditTests(unittest.TestCase):
    def test_weekend_publication_uses_trading_row_distance(self):
        k = k_frame()
        step_index = int(np.flatnonzero(k["date"] == pd.Timestamp("2020-06-02"))[0])
        add_bps_step(k, step_index)
        # Saturday before Tuesday: three calendar days, but only one trading row away.
        profit = pd.DataFrame({"pubDate": ["2020-05-30"]})
        summary, events = pit.audit_frames("sh.600000", k, profit)
        event = events.loc[events["metric"] == "bps"].iloc[0]
        self.assertTrue(bool(event["matched_publication"]))
        self.assertEqual(event["matched_pubDate"], pd.Timestamp("2020-05-30"))
        self.assertEqual(event["publication_gap_trading_days"], 1.0)
        self.assertEqual(summary["bps_matched"], 1)

    def test_publication_outside_trading_window_does_not_match(self):
        k = k_frame()
        step_index = int(np.flatnonzero(k["date"] == pd.Timestamp("2020-06-01"))[0])
        add_bps_step(k, step_index)
        # Wednesday before Monday is three trading rows earlier, outside [-2,+4].
        profit = pd.DataFrame({"pubDate": ["2020-05-27"]})
        summary, events = pit.audit_frames("sh.600000", k, profit)
        event = events.loc[events["metric"] == "bps"].iloc[0]
        self.assertFalse(bool(event["matched_publication"]))
        self.assertTrue(pd.isna(event["matched_pubDate"]))
        self.assertEqual(summary["bps_unmatched"], 1)

    def test_quarter_end_false_evidence_and_empty_profit(self):
        k = k_frame()
        step_index = int(np.flatnonzero(k["date"] == pd.Timestamp("2020-03-31"))[0])
        add_bps_step(k, step_index)
        summary, events = pit.audit_frames("sh.600000", k, pd.DataFrame())
        event = events.loc[events["metric"] == "bps"].iloc[0]
        self.assertTrue(bool(event["near_quarter_end"]))
        self.assertTrue(bool(event["quarter_end_unmatched"]))
        self.assertFalse(bool(event["matched_publication"]))
        self.assertEqual(summary["qe_unmatched"], 1)

    def test_publication_coverage_counts_only_dates_with_steps(self):
        k = k_frame()
        step_index = 100
        add_bps_step(k, step_index)
        step_date = k.loc[step_index, "date"]
        other_date = k.loc[step_index + 30, "date"]
        profit = pd.DataFrame({"pubDate": [step_date, other_date]})
        summary, _ = pit.audit_frames("sh.600000", k, profit)
        self.assertEqual(summary["n_pub"], 2)
        self.assertEqual(summary["pub_covered"], 1)

    def test_eps_ignores_nonpositive_pe_regime(self):
        k = k_frame()
        k["peTTM"] = -5.0
        summary, events = pit.audit_frames("sh.600000", k, pd.DataFrame())
        self.assertEqual(summary["eps_steps"], 0)
        self.assertTrue(events.empty)

    def test_stratified_sample_covers_all_strata_and_excludes_new_listings(self):
        rows = []
        eras = [("pre2018", "2010-01-04"), ("2018_2020", "2019-01-02"),
                ("2021_2024", "2022-01-04")]
        number = 0
        for exchange, prefix in (("SSE", "sh"), ("SZSE", "sz")):
            for _, ipo_date in eras:
                for status in (0, 1):
                    for _ in range(2):
                        number += 1
                        rows.append({"code": f"{prefix}.{number:06d}", "ipoDate": ipo_date,
                                     "status": status, "exchange": exchange})
        rows.append({"code": "sh.999999", "ipoDate": "2024-06-03",
                     "status": 1, "exchange": "SSE"})
        universe = pd.DataFrame(rows)
        sample = pit.select_sample(universe, n=12, seed=17)
        self.assertEqual(len(sample), 12)
        self.assertEqual(sample["audit_stratum"].nunique(), 12)
        self.assertEqual(sample.groupby("audit_stratum").size().max(), 1)
        self.assertNotIn("sh.999999", set(sample["code"]))

    def test_fewer_than_400_rows_is_not_audited(self):
        summary, events = pit.audit_frames("sz.000001", k_frame(399), pd.DataFrame())
        self.assertIsNone(summary)
        self.assertTrue(events.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
