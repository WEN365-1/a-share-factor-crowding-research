"""24 — 融资快照 45/90 自然日阈值的同口径敏感性检验。

只改变 ``load_margin_monthly`` 的 max_stale_days。形成月股票池、冻结权重、
T+1 可得性、缺失值处理、覆盖率、因子收益、控制变量和置换估计量均复用主分析。
45 天重建值先与既有 crowding.parquet 逐项核对，失败则不运行回归。

输出：
  data/clean/margin_staleness_45.parquet / _90.parquet
  reports/margin_staleness_diagnostics.csv
  reports/margin_staleness_robustness.csv
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from common import (CLEAN, REPORTS, formation_universe, get_logger, hac_ols,  # noqa: E402
                    placebo_pvalue, read_parquet, time_index, write_parquet)

crowd_mod = importlib.import_module("08_crowding")
regs = importlib.import_module("09_regressions")
log = get_logger("24_margin_staleness", "24_margin_staleness.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
MARGIN_COLS = ["crowd_margin", "crowd_margin_long", "level_margin", "cover_long", "cover_short"]


def build_membership(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ym, g in monthly.groupby("ym", observed=True):
        g = formation_universe(g)
        for factor in FACTORS:
            d = g.dropna(subset=[factor])
            if len(d) < crowd_mod.N_GROUPS * 5:
                continue
            try:
                d = d.assign(grp=pd.qcut(d[factor].rank(method="first"),
                                        crowd_mod.N_GROUPS, labels=False))
            except ValueError:
                continue
            rows.append(d[["ym", "code", "float_mv"]].assign(factor=factor, grp=d["grp"].values))
    return pd.concat(rows, ignore_index=True)


def assert_baseline_unchanged(rebuilt: pd.DataFrame, existing: pd.DataFrame) -> None:
    """主规格重构必须复现既有数值；不是只检查相关系数或显著性。"""
    a = rebuilt.copy()
    b = existing.copy()
    a["ym"] = pd.PeriodIndex(a["ym"], freq="M").astype(str)
    b["ym"] = pd.PeriodIndex(b["ym"], freq="M").astype(str)
    keys = ["ym", "factor"]
    a = a[keys + MARGIN_COLS].sort_values(keys).reset_index(drop=True)
    b = b[keys + MARGIN_COLS].sort_values(keys).reset_index(drop=True)
    if not a[keys].equals(b[keys]):
        raise AssertionError("45 天重建的月份/因子键与既有 crowding.parquet 不一致")
    for col in MARGIN_COLS:
        x = a[col].to_numpy(dtype=float)
        y = b[col].to_numpy(dtype=float)
        if not np.allclose(x, y, rtol=0, atol=1e-12, equal_nan=True):
            diff = np.abs(x - y)
            raise AssertionError(f"45 天重建改变 {col}: max abs diff={np.nanmax(diff)}")


def variant_panel(base: pd.DataFrame, crowd: pd.DataFrame) -> pd.DataFrame:
    drop = []
    for col in MARGIN_COLS + ["signal_date", "max_available_date", "n_long", "n_short"]:
        drop.extend((col, col + "_z"))
    keep = base.drop(columns=[c for c in drop if c in base.columns]).copy()
    crowd = crowd.copy()
    crowd["ym"] = pd.PeriodIndex(crowd["ym"], freq="M")
    out = keep.merge(crowd, on=["ym", "factor"], how="inner", validate="one_to_one")
    for col in MARGIN_COLS:
        out[col + "_z"] = (out[col] - out[col].mean()) / out[col].std()
    return out.sort_values(["ym", "factor"]).copy()


def bh_adjust(p_values) -> np.ndarray:
    """Benjamini--Hochberg 校正值：从最大 p 起向前取累积最小值。"""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("p_values 必须是一维非空有限数值")
    order = np.argsort(p)
    ranked = p[order]
    scaled = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    restored = np.empty(len(p))
    restored[order] = np.minimum(adjusted, 1.0)
    return restored


def main() -> None:
    monthly = read_parquet(CLEAN / "monthly.parquet")
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    monthly["code"] = monthly["code"].astype(str)
    membership = build_membership(monthly)
    calendar = read_parquet(CLEAN / "calendar.parquet")
    trading_dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"])).sort_values().unique()
    existing = read_parquet(CLEAN / "crowding.parquet")
    base = regs.build_panel()

    diagnostics = []
    panels = {}
    for threshold in (45, 90):
        log.info("构造 %s 天阈值", threshold)
        mg = crowd_mod.load_margin_monthly(trading_dates, max_stale_days=threshold)
        crowd = crowd_mod.build_margin_crowding(monthly, membership, mg)
        if threshold == 45:
            assert_baseline_unchanged(crowd, existing)
            log.info("45 天重建与既有五项融资字段逐项一致")
        write_parquet(crowd, CLEAN / f"margin_staleness_{threshold}.parquet")
        panels[threshold] = variant_panel(base, crowd)
        diagnostics.append({
            "max_stale_days": threshold,
            "raw_grid_rows": len(mg),
            "stale_excluded_rows": int(mg["stale_excluded"].sum()),
            "valid_margin_balance_rows": int(mg["margin_balance"].notna().sum()),
            "valid_d_mb_rows": int(mg["d_mb"].notna().sum()),
            "factor_month_rows": len(crowd),
            "mean_cover_long": crowd["cover_long"].mean(),
            "mean_cover_short": crowd["cover_short"].mean(),
            "mean_crowd_margin": crowd["crowd_margin"].mean(),
            "sd_crowd_margin": crowd["crowd_margin"].std(),
        })
    pd.DataFrame(diagnostics).to_csv(REPORTS / "margin_staleness_diagnostics.csv",
                                     index=False, encoding="utf-8-sig")

    rows = []
    for threshold, df in panels.items():
        measure = "crowd_margin_z"
        ctrl = regs.controls_for(df, "D")
        for horizon in (1, 3, 6, 12):
            lags = 12 if horizon == 12 else 6
            cols = [measure] + ctrl
            d = df.dropna(subset=cols + [f"fwd{horizon}"]).copy()
            r = hac_ols(d[f"fwd{horizon}"].to_numpy(), d[cols].to_numpy(), cols, lags=lags,
                        time=time_index(d["ym"]))
            pp, ndraw = placebo_pvalue(d, cols, measure, f"fwd{horizon}", r[measure]["t"],
                                       z_by="factor", hac_lags=lags)
            rows.append({"max_stale_days": threshold, "spec": "D", "horizon": horizon,
                         "return_type": "horizon_sum", "hac_lags": lags,
                         "beta": r[measure]["coef"], "se": r[measure]["se"],
                         "t": r[measure]["t"], "p_hac": r[measure]["p"],
                         "p_placebo": pp, "n_placebo": ndraw, "n": r["_n"],
                         "r2": r["_r2"]})
    out = pd.DataFrame(rows)
    # 每个阈值内部对四个持有期统一校正；45 天仍是主规格，90 天是敏感性而非新主检验。
    out["p_bh_4"] = np.nan
    out["p_bonferroni_4"] = np.nan
    for threshold, idx in out.groupby("max_stale_days").groups.items():
        p = out.loc[idx, "p_placebo"].to_numpy(dtype=float)
        out.loc[idx, "p_bh_4"] = bh_adjust(p)
        out.loc[idx, "p_bonferroni_4"] = np.minimum(p * len(p), 1.0)
    out.to_csv(REPORTS / "margin_staleness_robustness.csv", index=False, encoding="utf-8-sig")
    log.info("\n%s", out.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
