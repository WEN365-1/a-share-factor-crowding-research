"""Formation-return robustness tests; synthetic data only."""
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


formation = load_module("27_formation_robustness")


class FormationRobustnessTests(unittest.TestCase):
    def test_forward_window_starts_with_current_realisation_row(self):
        frame = pd.DataFrame({
            "ym": pd.period_range("2020-01", periods=15, freq="M"),
            "factor": "MOM",
            "ls_fwd1": np.arange(1, 16, dtype=float) / 100,
        })
        out = formation.add_return_windows(frame)
        self.assertAlmostEqual(out.loc[0, "fwd1"], 0.01)
        self.assertAlmostEqual(out.loc[0, "fwd3"], 0.06)
        self.assertAlmostEqual(out.loc[10, "fwd3"], 0.36)
        self.assertTrue(pd.isna(out.loc[13, "fwd3"]))

    def test_past_controls_use_only_prior_realisations(self):
        frame = pd.DataFrame({
            "ym": pd.period_range("2020-01", periods=15, freq="M"),
            "factor": "MOM",
            "ls_fwd1": np.arange(1, 16, dtype=float) / 100,
        })
        out = formation.add_return_windows(frame)
        self.assertTrue(pd.isna(out.loc[0, "past1"]))
        self.assertAlmostEqual(out.loc[1, "past1"], 0.01)
        self.assertAlmostEqual(out.loc[6, "past6"], 0.21)
        self.assertAlmostEqual(out.loc[12, "past12"], 0.78)

    def test_replacement_preserves_predictors_and_input(self):
        base = pd.DataFrame({
            "ym": pd.period_range("2020-01", periods=3, freq="M"),
            "factor": "MOM",
            "ls_fwd1": [0.01, 0.02, 0.03],
            "crowd_margin_z": [1.0, 2.0, 3.0],
            "FE_VAL": [0.0, 0.0, 0.0],
        })
        original = base.copy(deep=True)
        alt = pd.DataFrame({
            "ym": ["2020-01", "2020-02", "2020-03"], "factor": "MOM",
            "alternative": [0.10, 0.20, 0.30],
        })
        out = formation.replace_returns(base, alt, "alternative")
        pd.testing.assert_frame_equal(base, original)
        pd.testing.assert_series_equal(out["crowd_margin_z"], base["crowd_margin_z"])
        self.assertEqual(out["ls_fwd1"].tolist(), [0.10, 0.20, 0.30])
        self.assertNotIn("alternative", out.columns)

    def test_replacement_rejects_duplicate_alternative_rows(self):
        base = pd.DataFrame({
            "ym": [pd.Period("2020-01", freq="M")], "factor": "MOM",
            "ls_fwd1": [0.01], "crowd_margin_z": [1.0],
        })
        alt = pd.DataFrame({
            "ym": ["2020-01", "2020-01"], "factor": "MOM",
            "alternative": [0.10, 0.20],
        })
        with self.assertRaises(ValueError):
            formation.replace_returns(base, alt, "alternative")


if __name__ == "__main__":
    unittest.main(verbosity=2)
