"""长期回归 HAC 口径的轻量回归测试；不读取或修改研究数据。"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
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


regs = load_module("09_regressions")
ext = load_module("16_extension_analyses")
flow = load_module("18_flow_proxy")
robustness = load_module("11_robustness")
faithful = load_module("20_comomentum_faithful_tests")


def fake_result(names):
    return {name: {"coef": 0.1, "se": 0.2, "t": 0.5, "p": 0.6} for name in names} | {
        "_n": 40, "_r2": 0.1,
    }


class HacHorizonTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "ym": pd.period_range("2020-01", periods=40, freq="M"),
            "factor": "MOM",
            "x": np.arange(40, dtype=float),
            "y": np.arange(40, dtype=float) / 100,
        })

    def check_fit_row(self, module, horizon, expected_lags):
        hac_calls, placebo_calls = [], []

        def fake_hac(y, x, cols, lags=6):
            hac_calls.append(lags)
            return fake_result(cols)

        def fake_placebo(d, cols, measure, y, obs_t, **kwargs):
            placebo_calls.append(kwargs.get("hac_lags"))
            return 0.5, 10

        with patch.object(module, "hac_ols", fake_hac), patch.object(
                module, "placebo_pvalue", fake_placebo):
            row = module.fit_row(self.frame, ["x"], "x", "y", {"horizon": horizon})
        self.assertEqual(hac_calls, [expected_lags])
        self.assertEqual(placebo_calls, [expected_lags])
        self.assertEqual(row["hac_lags"], expected_lags)

    def test_extension_fit_row_matches_observed_and_placebo_lags(self):
        for horizon, expected in ((6, 6), (12, 12)):
            with self.subTest(horizon=horizon):
                self.check_fit_row(ext, horizon, expected)

    def test_flow_fit_row_matches_observed_and_placebo_lags(self):
        for horizon, expected in ((6, 6), (12, 12)):
            with self.subTest(horizon=horizon):
                self.check_fit_row(flow, horizon, expected)

    def test_main_regression_matches_observed_and_placebo_lags(self):
        for horizon, expected in ((6, 6), (12, 12)):
            hac_calls, placebo_calls = [], []

            def fake_hac(y, x, cols, lags=6):
                hac_calls.append(lags)
                return fake_result(cols)

            def fake_placebo(d, cols, measure, y, obs_t, **kwargs):
                placebo_calls.append(kwargs.get("hac_lags"))
                return 0.5, 10

            with self.subTest(horizon=horizon), patch.object(
                    regs, "hac_ols", fake_hac), patch.object(
                    regs, "placebo_pvalue", fake_placebo):
                _, _, _, lags = regs.infer_horizon(self.frame, ["x"], "x", "y", horizon)
            self.assertEqual(lags, expected)
            self.assertEqual(hac_calls, [expected])
            self.assertEqual(placebo_calls, [expected])

    def test_weekly_analogue_matches_observed_and_placebo_lags(self):
        frame = self.frame.rename(columns={"x": "m_z", "y": "fwd6"}).copy()
        frame["fwd12"] = frame["fwd6"]
        for horizon, expected in ((6, 6), (12, 12)):
            hac_calls, placebo_calls = [], []

            def fake_hac(y, x, cols, lags=6):
                hac_calls.append(lags)
                return fake_result(cols)

            def fake_placebo(d, cols, measure, y, obs_t, **kwargs):
                placebo_calls.append(kwargs.get("hac_lags"))
                return 0.5, 10

            with self.subTest(horizon=horizon), patch.object(
                    faithful, "hac_ols", fake_hac), patch.object(
                    faithful, "placebo_pvalue", fake_placebo):
                row = faithful.estimate(frame, [], "m", horizon, "synthetic")
            self.assertEqual(row["hac_lags"], expected)
            self.assertEqual(hac_calls, [expected])
            self.assertEqual(placebo_calls, [expected])

    def test_interaction_null_uses_requested_lags(self):
        frame = self.frame.iloc[:30].copy()
        frame["high_comom"] = (np.arange(30) >= 15).astype(float)
        frame["inter"] = frame["x"] * frame["high_comom"]
        calls = []

        def fake_hac(y, x, cols, lags=6):
            calls.append(lags)
            return {"inter": {"t": 0.5}}

        with patch.object(ext, "hac_ols", fake_hac):
            null = ext.placebo_null_interaction(
                frame, ["x", "high_comom", "inter"], "x", "inter", "y", hac_lags=12)
        self.assertEqual(len(null), 29)
        self.assertEqual(set(calls), {12})

    def test_robustness_fit_drops_preexisting_fixed_effects(self):
        frame = pd.DataFrame({
            "ym": pd.period_range("2020-01", periods=40, freq="M"),
            "factor": ["A"] * 20 + ["B"] * 20,
            "x": np.arange(40, dtype=float),
            "y": np.arange(40, dtype=float) / 100,
            "FE_B": [0.0] * 20 + [1.0] * 20,
        })
        shapes = []

        def fake_hac(y, x, cols, lags=6):
            shapes.append((x.shape, list(cols)))
            return fake_result(cols)

        with patch.object(robustness, "hac_ols", fake_hac), patch.object(
                robustness, "placebo_pvalue", return_value=(0.5, 10)):
            robustness.fit(frame, "x", y="y")
        self.assertEqual(shapes, [((40, 2), ["x", "FE_B"])])


if __name__ == "__main__":
    unittest.main(verbosity=2)
