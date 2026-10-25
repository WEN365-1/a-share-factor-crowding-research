"""多种子 comomentum 审计的抽样与面板替换测试。"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def load(stem: str):
    path = PROJECT / "code" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crowd = load("08_crowding")
        cls.audit = load("30_comomentum_sampling")

    def test_seed_reproduces_and_sampling_is_without_replacement(self):
        rng = np.random.default_rng(17)
        Res = pd.DataFrame(rng.normal(size=(90, 8)), columns=list("abcdefgh"))
        # 人为低于 MIN_OBS_WINDOW 不影响这里直接传入的残差；只检验公共抽样函数。
        kwargs = dict(sample_per_leg=4, min_per_leg=4, min_pairs=20)
        a = self.crowd.sampled_leg_correlation(Res, list(Res.columns), np.random.default_rng(99), **kwargs)
        b = self.crowd.sampled_leg_correlation(Res, list(Res.columns), np.random.default_rng(99), **kwargs)
        self.assertEqual(a, b)
        self.assertEqual(a["n_sampled"], 4)
        self.assertEqual(a["n_pairs"], 6)

    def test_seed_change_can_change_sample_but_not_algorithm(self):
        rng = np.random.default_rng(17)
        Res = pd.DataFrame(rng.normal(size=(90, 8)), columns=list("abcdefgh"))
        kwargs = dict(sample_per_leg=4, min_per_leg=4, min_pairs=20)
        a = self.crowd.sampled_leg_correlation(Res, list(Res.columns), np.random.default_rng(1), **kwargs)
        b = self.crowd.sampled_leg_correlation(Res, list(Res.columns), np.random.default_rng(2), **kwargs)
        self.assertNotEqual(a["correlation"], b["correlation"])
        self.assertEqual(a["n_pairs"], b["n_pairs"])

    def test_monthly_summary_does_not_stack_seeds(self):
        draws = pd.DataFrame({"ym": ["2020-01"] * 3, "factor": ["MOM"] * 3,
                              "seed": [1, 2, 3], "comomentum": [-1.0, 1.0, 1.0]})
        out = self.audit.summarize_draws(draws)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "n_draws"], 3)
        self.assertAlmostEqual(out.loc[0, "sign_stability"], 2 / 3)

    def test_seed_replace_restandardizes_and_preserves_row_count(self):
        base = pd.DataFrame({"ym": pd.period_range("2020-01", periods=2, freq="M"),
                             "factor": ["MOM", "MOM"], "comomentum": [8.0, 9.0],
                             "comomentum_z": [100.0, 100.0], "x": [1.0, 2.0]})
        draw = pd.DataFrame({"ym": base["ym"], "factor": ["MOM", "MOM"],
                             "comomentum": [1.0, 3.0]})
        out = self.audit.replace_seed_measure(base, draw)
        self.assertEqual(len(out), len(base))
        self.assertEqual(out["x"].tolist(), [1.0, 2.0])
        self.assertAlmostEqual(out["comomentum_z"].mean(), 0.0)
        self.assertAlmostEqual(out["comomentum_z"].std(ddof=1), 1.0)
        self.assertTrue(np.allclose(out["comomentum_z"], [-1 / np.sqrt(2), 1 / np.sqrt(2)]))

    def test_seed_list_is_fixed_and_contains_production_seed(self):
        self.assertEqual(len(self.audit.SEEDS), 20)
        self.assertEqual(len(set(self.audit.SEEDS)), 20)
        self.assertEqual(self.audit.SEEDS[0], self.crowd.SEED)
        self.assertEqual(self.audit.SEEDS, tuple(range(self.crowd.SEED, self.crowd.SEED + 20)))


if __name__ == "__main__":
    unittest.main()
