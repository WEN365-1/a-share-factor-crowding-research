"""23_lp_second_moments.py 的时间对齐与独立数值核验。"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("FIN_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "code"))
SPEC = importlib.util.spec_from_file_location("second_moments_23", ROOT / "code" / "23_lp_second_moments.py")
M23 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M23)


class SecondMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [
            ROOT / "data" / "clean" / "factor_returns.parquet",
            ROOT / "data" / "clean" / "crowding.parquet",
            ROOT / "data" / "clean" / "factor_daily.parquet",
        ]
        missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "Provider-derived inputs are not redistributed in the route-B package; "
                "supply them lawfully to enable these tests: " + ", ".join(missing)
            )
        cls.out = M23.build_outcomes()
        cls.daily = pd.read_parquet(ROOT / "data" / "clean" / "factor_daily.parquet")
        cls.daily["date"] = pd.to_datetime(cls.daily["date"])
        cls.daily["ym"] = pd.PeriodIndex(cls.daily["ym"], freq="M")

    def test_compound_forward_starts_at_current_forward_return(self):
        s = pd.Series([0.01, 0.02, 0.03, 0.04])
        self.assertAlmostEqual(M23.compound_forward(s, 2).iloc[0], 1.01 * 1.02 - 1)

    def test_unique_panel_keys(self):
        self.assertFalse(self.out.duplicated(["factor", "ym", "horizon"]).any())

    def test_twelve_month_daily_moments_match_raw_days(self):
        r = self.out[(self.out.factor == "MOM") & (self.out.horizon == 12)
                     & self.out.daily_vol.notna()].iloc[0]
        months = pd.period_range(r.ym, r.ym + 11, freq="M")
        x = self.daily[(self.daily.factor == "MOM") & self.daily.ym.isin(months)]
        self.assertEqual(len(x), r.n_days)
        self.assertAlmostEqual(x.ls_fwd1.std(ddof=0), r.daily_vol, places=6)
        self.assertAlmostEqual((x.ls_fwd1 < -0.01).mean(), r.daily_bad_share, places=12)

    def test_cross_month_week_compounded_once(self):
        r = self.out[(self.out.factor == "MOM") & (self.out.horizon == 12)
                     & self.out.weekly_skew.notna()].iloc[0]
        months = pd.period_range(r.ym, r.ym + 11, freq="M")
        x = self.daily[(self.daily.factor == "MOM") & self.daily.ym.isin(months)].copy()
        x["week"] = x.date.dt.to_period("W-FRI")
        w = x.groupby("week").ls_fwd1.apply(lambda s: (1 + s).prod() - 1)
        self.assertEqual(len(w), r.n_weeks)
        skew = ((w - w.mean()) ** 3).mean() / w.std(ddof=0) ** 3
        self.assertAlmostEqual(skew, r.weekly_skew, places=5)
        self.assertAlmostEqual((w < -0.05).mean(), r.weekly_bad_share, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
