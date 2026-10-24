"""22_comomentum_factor_slopes.py —— 12 个月因子特定斜率与 MOM 特异性。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, get_logger, now_local, regression_frame  # noqa: E402

log = get_logger("22_comomentum_factor_slopes", "22_comomentum_factor_slopes.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
MEASURES = ["comomentum", "comomentum_lp", "comomentum_lp_screen"]


def load_21():
    path = Path(__file__).resolve().with_name("21_comomentum_robustness.py")
    spec = importlib.util.spec_from_file_location("comomentum_21", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module21 = load_21()
    df, controls = module21.build_panel()
    rows = []
    for factor in FACTORS:
        frame = regression_frame(df[df["factor"] == factor])
        for return_type in ["sum", "compound"]:
            for measure in MEASURES:
                row = module21.estimate(frame, controls, measure, factor, return_type, 12)
                rows.append(row)
    out = pd.DataFrame(rows)
    out["p_bonferroni_6"] = out.groupby(["measure", "return_type"])["p_placebo"].transform(
        lambda s: (6 * s).clip(upper=1.0))
    out["p_bh_6"] = np.nan
    for _, idx in out.groupby(["measure", "return_type"]).groups.items():
        out.loc[idx, "p_bh_6"] = multipletests(out.loc[idx, "p_placebo"], method="fdr_bh")[1]
    out.to_csv(REPORTS / "comomentum_h12_factor_slopes.csv", index=False, encoding="utf-8-sig")
    log.info("12 个月因子特定斜率 | %s\n%s", now_local(),
             out[["sample", "measure", "return_type", "beta", "t", "p_placebo", "p_bh_6", "n"]]
             .round(5).to_string(index=False))


if __name__ == "__main__":
    main()
