"""Driscoll–Kraay 标准误的单元测试；只用合成数据，不读取研究数据。"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import statsmodels.api as sm

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
os.environ["FIN_ROOT"] = str(PROJECT)
sys.path.insert(0, str(CODE))


def load_common():
    spec = importlib.util.spec_from_file_location("common_for_dk_test", CODE / "common.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


common = load_common()


def make_panel(seed: int = 20261025, n_months: int = 120, n_rows_per_month: int = 6):
    rng = np.random.default_rng(seed)
    time = np.repeat(np.arange(n_months), n_rows_per_month)
    group = np.tile(np.arange(n_rows_per_month), n_months)
    n = len(time)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    fe = (group >= 1).astype(float)
    month_shock = np.repeat(rng.normal(size=n_months), n_rows_per_month)
    eps = 0.7 * month_shock + 0.3 * rng.normal(size=n)
    y = 0.5 * x1 - 0.2 * x2 + 0.3 * fe + eps
    X = np.column_stack([x1, x2, fe])
    return y, X, ["x1", "x2", "fe"], time, group


def statsmodels_groupsum(y, X, time, lags):
    Xc = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    return sm.OLS(y, Xc).fit(
        cov_type="hac-groupsum",
        cov_kwds={"time": np.asarray(time), "maxlags": lags, "use_correction": False},
    )


class DriscollKraayTests(unittest.TestCase):
    def test_matches_statsmodels_hac_groupsum(self):
        """手写 DK 与 statsmodels hac-groupsum（无小样本修正）的标准误一致。"""
        y, X, names, time, _ = make_panel()
        lags = 6
        ours = common.hac_ols(y, X, names, lags=lags, time=time)
        ref = statsmodels_groupsum(y, X, time, lags)
        for i, name in enumerate(["const"] + names):
            expected = float(ref.bse[i])
            actual = ours[name]["se"]
            rel = abs(actual - expected) / max(abs(expected), 1e-300)
            self.assertLess(rel, 1e-8, f"{name}: ours={actual:.17g} ref={expected:.17g} rel={rel:.3e}")

    def test_single_row_per_period_equals_newey_west(self):
        """每个 time 只有一行时，DK 退化为普通 Newey–West（同阶数、无修正）。"""
        rng = np.random.default_rng(7)
        n_months = 120
        time = np.arange(n_months)
        x1 = rng.normal(size=n_months)
        x2 = rng.normal(size=n_months)
        shock = np.zeros(n_months)
        for t in range(1, n_months):
            shock[t] = 0.6 * shock[t - 1] + rng.normal()
        y = 0.4 * x1 - 0.1 * x2 + shock
        X = np.column_stack([x1, x2])
        names = ["x1", "x2"]
        lags = 6
        ours = common.hac_ols(y, X, names, lags=lags, time=time)
        Xc = sm.add_constant(X, has_constant="add")
        ref = sm.OLS(y, Xc).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
        for i, name in enumerate(["const"] + names):
            self.assertAlmostEqual(ours[name]["se"] / float(ref.bse[i]), 1.0, places=10)

    def test_row_order_within_month_does_not_matter(self):
        """同一个月内打乱行顺序，DK 的系数与标准误都不变。"""
        y, X, names, time, _ = make_panel(seed=11)
        lags = 12
        base = common.hac_ols(y, X, names, lags=lags, time=time)
        rng = np.random.default_rng(99)
        # 只在每个月内部打乱，保留「每个月连续成块」的排列。
        blocks = [np.arange(i * 6, (i + 1) * 6) for i in range(120)]
        order = np.concatenate([rng.permutation(b) for b in blocks])
        shuffled = common.hac_ols(y[order], X[order], names, lags=lags, time=time[order])
        for key in ["const"] + names:
            self.assertAlmostEqual(shuffled[key]["coef"], base[key]["coef"], places=12)
            self.assertAlmostEqual(shuffled[key]["se"], base[key]["se"], places=12)

    def test_coefficients_identical_to_plain_hac_branch(self):
        """加入 time 只改变标准误，系数与 time=None 时完全相同。"""
        y, X, names, time, _ = make_panel(seed=2026)
        plain = common.hac_ols(y, X, names, lags=6)
        dk = common.hac_ols(y, X, names, lags=6, time=time)
        for key in ["const"] + names:
            self.assertAlmostEqual(dk[key]["coef"], plain[key]["coef"], places=13)
        self.assertEqual(dk["_n"], plain["_n"])
        self.assertAlmostEqual(dk["_r2"], plain["_r2"], places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
