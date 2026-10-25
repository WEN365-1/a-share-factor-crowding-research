"""
19_comomentum_faithful.py —— Lou–Polk comomentum 的忠实核心口径对照

与 08 的日频抽样两两相关版本分开保存，不覆盖 P0 结果。

LP core（本地数据可实现的原文核心构造）：
  * 52 个 Friday-to-Friday 周收益；
  * 个股收益先减去行业等权周收益，再对周频 FF3 做窗口内时序回归；
  * 每端计算个股残差与“剔除该股的等权 decile 组合”的相关，再对个股平均；
  * 多空两端等权平均，不抽样股票。

LP screen 是额外的中国市场微观结构适配：在每月形成股票池中剔除收盘价低于
5 元或处于最小流通市值十分位的股票，再重新十分位分组。它不是美股规则的
完全复制；行业分类仍是当前快照，FF3 仍由本项目的 A 股样本构造。

同时输出六个对称分解版本：lp、lp_no_ind、lp_pair、lp_full52、lp_screen、
lp_screen_full52，分别识别行业调整、相关公式、完整 52 周和本地过滤的影响。
21 进一步比较月收益和/复利与 HAC(6)/HAC(12)，22 报告因子特定斜率。

固定规则（在看到预测结果前确定）：52 周窗口；个股至少 39 周有效；个股与
剔除自身组合至少 30 周配对；组合每周至少 5 只有效股票；每端至少 15 只。
"""
from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, REPORTS, Gate, formation_universe, get_logger,  # noqa: E402
                    now_local, read_parquet, write_parquet)

log = get_logger("19_comomentum_faithful", "19_comomentum_faithful.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
N_GROUPS = 10
FORM_WEEKS = 52
MIN_STOCK_WEEKS = 39
MIN_PAIRS_WEEKS = 30
MIN_OTHER_STOCKS = 5
MIN_PER_LEG = 15
PRICE_MIN = 5.0


def load_08():
    path = Path(__file__).resolve().with_name("08_crowding.py")
    spec = importlib.util.spec_from_file_location("crowding_08", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """把日收益复利聚合为周五标签的周收益；观测行内未知收益不当作零。"""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["code"] = d["code"].astype(str)
    d = d.sort_values(["code", "date"])
    d["week"] = d["date"].dt.to_period("W-FRI")
    d["gross"] = 1.0 + d["ret"]
    out = (d.groupby(["code", "week"], observed=True)
           .agg(gross=("gross", "prod"), n_ret=("ret", "count"),
                n_rows=("ret", "size"), date=("date", "last"),
                close=("close", "last"), float_mv=("float_mv", "last"),
                industry=("industry", "last"))
           .reset_index())
    out["ret_week"] = out["gross"] - 1.0
    out.loc[(out["n_ret"] == 0) | (out["n_ret"] < out["n_rows"]), "ret_week"] = np.nan
    out["week_end"] = out["week"].dt.end_time.dt.normalize()
    out["ym"] = out["date"].dt.to_period("M")
    return out.drop(columns=["gross", "n_ret", "n_rows"])


def load_weekly() -> pd.DataFrame:
    files = sorted((CLEAN / "panel").rglob("*.parquet"))
    parts = []
    cols = ["date", "code", "ret", "close", "float_mv", "industry"]
    for f in files:
        parts.append(aggregate_weekly(pd.read_parquet(f, columns=cols)))
    if not parts:
        raise ValueError("没有可用于周收益聚合的面板文件")
    out = pd.concat(parts, ignore_index=True)
    # 跨年周会被年度分片切成两段；必须再次复利合并，不能把两段当成同一
    # (week, code) 的重复观测。任一片段收益未知，则完整周收益仍未知。
    out = out.sort_values(["code", "date"])
    out["gross"] = 1.0 + out["ret_week"]
    out = (out.groupby(["code", "week"], observed=True)
           .agg(gross=("gross", "prod"), n_valid=("ret_week", "count"),
                n_parts=("ret_week", "size"), date=("date", "last"),
                close=("close", "last"), float_mv=("float_mv", "last"),
                industry=("industry", "last"))
           .reset_index())
    out["ret_week"] = out["gross"] - 1.0
    out.loc[(out["n_valid"] == 0) | (out["n_valid"] < out["n_parts"]), "ret_week"] = np.nan
    out["week_end"] = out["week"].dt.end_time.dt.normalize()
    out["ym"] = out["date"].dt.to_period("M")
    out = out.drop(columns=["gross", "n_valid", "n_parts"])
    if out.duplicated(["week_end", "code"]).any():
        raise ValueError("跨年周合并后仍存在重复股票周")
    out = out.sort_values(["week_end", "code"]).reset_index(drop=True)
    log.info("周收益面板：%s 行 | %s 只股票 | %s 周",
             f"{len(out):,}", out["code"].nunique(), out["week_end"].nunique())
    return out


def microstructure_screen(g: pd.DataFrame, price_min: float = PRICE_MIN) -> pd.Series:
    """本地适配过滤：低价或最小流通市值十分位；缺失价格也排除。"""
    size_cut = g["float_mv"].quantile(0.10)
    return (g["close"].notna() & (g["close"] >= price_min)
            & (g["float_mv"] > size_cut))


def assign_deciles(g: pd.DataFrame, ym: pd.Period) -> pd.DataFrame:
    rows = []
    for f in FACTORS:
        d = g.dropna(subset=[f])
        if len(d) < N_GROUPS * 5:
            continue
        try:
            grp = pd.qcut(d[f].rank(method="first"), N_GROUPS, labels=False)
        except ValueError:
            continue
        rows.append(d[["ym", "code", "float_mv"]].assign(factor=f, grp=grp.to_numpy()))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_memberships(monthly: pd.DataFrame, close_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = monthly.copy()
    monthly["ym"] = pd.PeriodIndex(monthly["ym"], freq="M")
    monthly["code"] = monthly["code"].astype(str)
    base_rows, screen_rows = [], []
    for ym, g0 in monthly.groupby("ym", observed=True):
        g = formation_universe(g0).merge(close_map, on=["code", "ym"], how="left")
        b = assign_deciles(g, ym)
        if len(b):
            base_rows.append(b)
        s = assign_deciles(g.loc[microstructure_screen(g)].copy(), ym)
        if len(s):
            screen_rows.append(s)
    base = pd.concat(base_rows, ignore_index=True)
    screen = pd.concat(screen_rows, ignore_index=True)
    log.info("LP 分组归属：基准 %s 行；微观结构过滤 %s 行",
             f"{len(base):,}", f"{len(screen):,}")
    return base, screen


def compound_ff3_weekly(ff3_daily: pd.DataFrame) -> pd.DataFrame:
    f = ff3_daily.copy()
    f.index = pd.to_datetime(f.index)
    f["week"] = f.index.to_period("W-FRI")
    gross = 1.0 + f[["MKT", "SMB", "HML"]]
    w = gross.groupby(f["week"], observed=True).prod(min_count=1) - 1.0
    w.index = w.index.end_time.normalize()
    w.index.name = "week_end"
    return w.sort_index()


def weekly_return_matrix(weekly: pd.DataFrame, ff3_weekly: pd.DataFrame,
                         industry_adjust: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 week × stock 收益矩阵；行业调整使用行业等权周收益。"""
    d = weekly.copy()
    if industry_adjust:
        ind = (weekly.dropna(subset=["industry", "ret_week"])
               .groupby(["week_end", "industry"], observed=True)["ret_week"]
               .mean().rename("ind_ret").reset_index())
        d = weekly.merge(ind, on=["week_end", "industry"], how="left")
        d["ret_input"] = d["ret_week"] - d["ind_ret"]
    else:
        d["ret_input"] = d["ret_week"]
    R = d.pivot(index="week_end", columns="code", values="ret_input").astype("float32")
    X = ff3_weekly.reindex(R.index).astype("float64")
    valid = X.notna().all(axis=1)
    R, X = R.loc[valid], X.loc[valid]
    log.info("LP 收益矩阵（行业调整=%s）：%s 周 × %s 只股票",
             industry_adjust, R.shape[0], R.shape[1])
    return R, X


def window_residuals(R: pd.DataFrame, X: pd.DataFrame, lo: int, hi: int,
                     min_obs: int = MIN_STOCK_WEEKS) -> pd.DataFrame:
    Rw = R.iloc[lo:hi]
    Xw = np.column_stack([np.ones(len(Rw)), X.iloc[lo:hi].to_numpy(dtype="float64")])
    A = Rw.to_numpy(dtype="float64")
    ok = np.isfinite(A)
    n_obs = ok.sum(axis=0)
    resid = np.full(A.shape, np.nan, dtype="float32")
    full = n_obs == A.shape[0]
    if full.any():
        cols = np.flatnonzero(full)
        beta, *_ = np.linalg.lstsq(Xw, A[:, cols], rcond=None)
        resid[:, cols] = (A[:, cols] - Xw @ beta).astype("float32")
    for c in np.flatnonzero((~full) & (n_obs >= min_obs)):
        m = ok[:, c]
        b, *_ = np.linalg.lstsq(Xw[m], A[m, c], rcond=None)
        resid[m, c] = (A[m, c] - Xw[m] @ b).astype("float32")
    return pd.DataFrame(resid, index=Rw.index, columns=Rw.columns)


def formation_window_indices(index: pd.DatetimeIndex, signal_date: pd.Timestamp,
                             window: int = FORM_WEEKS) -> tuple[int, int] | None:
    """仅使用周五标签不晚于信号日的完整周，排除信号日后的周收益。"""
    hi = int(np.searchsorted(index.to_numpy(), np.datetime64(signal_date), side="right"))
    lo = hi - window
    return (lo, hi) if lo >= 0 else None


def leave_one_out_correlations(Res: pd.DataFrame, codes: list[str],
                               min_stock_weeks: int = MIN_STOCK_WEEKS,
                               min_pairs_weeks: int = MIN_PAIRS_WEEKS,
                               min_other_stocks: int = MIN_OTHER_STOCKS,
                               min_per_leg: int = MIN_PER_LEG) -> dict:
    """原文式(1)–(2)：corr(e_i, mean_{j≠i}e_j)，再对 i 等权平均。"""
    cols = [c for c in dict.fromkeys(codes) if c in Res.columns]
    if len(cols) < min_per_leg:
        return {"comomentum": np.nan, "n": len(cols), "mean_weeks": np.nan,
                "mean_pair_weeks": np.nan}
    A = Res[cols].to_numpy(dtype="float64")
    valid = np.isfinite(A)
    n_obs = valid.sum(axis=0)
    keep = n_obs >= min_stock_weeks
    A = A[:, keep]
    n_obs = n_obs[keep]
    if A.shape[1] < min_per_leg:
        return {"comomentum": np.nan, "n": int(keep.sum()), "mean_weeks": np.nan,
                "mean_pair_weeks": np.nan}
    valid = np.isfinite(A)
    total_n = valid.sum(axis=1)
    total_sum = np.nansum(A, axis=1)
    corrs, pair_weeks = [], []
    for j in range(A.shape[1]):
        vi = valid[:, j]
        other_n = total_n - vi
        other_sum = total_sum - np.where(vi, A[:, j], 0.0)
        m = vi & (other_n >= min_other_stocks)
        if m.sum() < min_pairs_weeks:
            continue
        x = A[m, j]
        p = other_sum[m] / other_n[m]
        if np.std(x) <= 0 or np.std(p) <= 0:
            continue
        corrs.append(float(np.corrcoef(x, p)[0, 1]))
        pair_weeks.append(int(m.sum()))
    return {"comomentum": float(np.mean(corrs)) if corrs else np.nan,
            "n": int(A.shape[1]),
            "mean_weeks": float(np.mean(n_obs)),
            "mean_pair_weeks": float(np.mean(pair_weeks)) if pair_weeks else np.nan}


def pairwise_correlations(Res: pd.DataFrame, codes: list[str],
                          min_stock_weeks: int = MIN_STOCK_WEEKS,
                          min_pairs_weeks: int = MIN_PAIRS_WEEKS,
                          min_per_leg: int = MIN_PER_LEG) -> dict:
    """全两两相关对照；股票有效周数过滤与 LP core 完全一致。"""
    cols = [c for c in dict.fromkeys(codes) if c in Res.columns]
    if len(cols) < min_per_leg:
        return {"comomentum": np.nan, "n": len(cols), "mean_weeks": np.nan,
                "mean_pair_weeks": np.nan}
    sub = Res[cols]
    n_obs = sub.notna().sum(axis=0)
    keep = n_obs[n_obs >= min_stock_weeks].index
    sub = sub[keep]
    if len(keep) < min_per_leg:
        return {"comomentum": np.nan, "n": len(keep), "mean_weeks": np.nan,
                "mean_pair_weeks": np.nan}
    C = sub.corr(min_periods=min_pairs_weeks).to_numpy()
    counts = sub.notna().astype("int32").to_numpy()
    pair_n = counts.T @ counts
    iu = np.triu_indices_from(C, k=1)
    values, pair_weeks = C[iu], pair_n[iu]
    valid = np.isfinite(values)
    return {"comomentum": float(values[valid].mean()) if valid.any() else np.nan,
            "n": int(len(keep)), "mean_weeks": float(n_obs[keep].mean()),
            "mean_pair_weeks": float(pair_weeks[valid].mean()) if valid.any() else np.nan}


def membership_index(membership: pd.DataFrame) -> dict:
    return {ym: {f: gg for f, gg in g.groupby("factor", observed=True)}
            for ym, g in membership.groupby("ym", observed=True)}


def diagnostics(new: pd.DataFrame) -> pd.DataFrame:
    old = read_parquet(CLEAN / "crowding.parquet")[["ym", "factor", "comomentum"]].copy()
    old["ym"] = old["ym"].astype(str)
    d = new.merge(old, on=["ym", "factor"], how="left")
    rows = []
    measures = [c for c in new.columns if c.startswith("comomentum_")]
    for measure in measures:
        for f, g in d.groupby("factor", observed=True):
            s = g.dropna(subset=[measure, "comomentum"])
            if len(s) >= 12:
                rows.append({"measure": measure, "factor": f, "n": len(s),
                             "pearson_with_daily": s[measure].corr(s["comomentum"]),
                             "spearman_with_daily": s[measure].corr(s["comomentum"], method="spearman"),
                             "mean": s[measure].mean(), "sd": s[measure].std()})
        s = d.dropna(subset=[measure, "comomentum"])
        rows.append({"measure": measure, "factor": "POOLED", "n": len(s),
                     "pearson_with_daily": s[measure].corr(s["comomentum"]),
                     "spearman_with_daily": s[measure].corr(s["comomentum"], method="spearman"),
                     "mean": s[measure].mean(), "sd": s[measure].std()})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "comomentum_faithful_diagnostics.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    gate = Gate(log)
    log.info("开始 LP comomentum 对照 | %s", now_local())
    monthly = read_parquet(CLEAN / "monthly.parquet")
    # 先用与 08 相同的日频 FF3，再复利到周频，避免跨月周收益使用错误分组。
    module08 = load_08()
    daily = load_weekly()  # 周面板同时提供月末价格，供独立微观结构过滤使用
    close_map = (daily.sort_values("date").groupby(["code", "ym"], observed=True)
                 [["close"]].last().reset_index())
    base_membership, screen_membership = build_memberships(monthly, close_map)
    del monthly, close_map
    gc.collect()

    daily08 = module08.load_daily()
    ff3_daily = module08.build_daily_ff3(daily08, base_membership)
    del daily08, module08
    gc.collect()
    ff3_weekly = compound_ff3_weekly(ff3_daily)
    del ff3_daily
    matrices = {
        "ind": weekly_return_matrix(daily, ff3_weekly, industry_adjust=True),
        "raw": weekly_return_matrix(daily, ff3_weekly, industry_adjust=False),
    }
    del daily, ff3_weekly
    gc.collect()

    cal = read_parquet(CLEAN / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    cal["ym"] = cal["date"].dt.to_period("M")
    month_ends = sorted(cal.groupby("ym", observed=True)["date"].max().tolist())
    bases = {"base": membership_index(base_membership),
             "screen": membership_index(screen_membership)}
    del base_membership, screen_membership
    gc.collect()
    # 完整分解，而非只报告支持某一结论的口径。
    configs = [
        ("lp", "base", "ind", "loo", 39),
        ("lp_no_ind", "base", "raw", "loo", 39),
        ("lp_pair", "base", "ind", "pair", 39),
        ("lp_full52", "base", "ind", "loo", 52),
        ("lp_screen", "screen", "ind", "loo", 39),
        ("lp_screen_full52", "screen", "ind", "loo", 52),
    ]

    rows = []
    reference_index = matrices["ind"][0].index
    for t in month_ends:
        if not gate.ok():
            raise RuntimeError("运行闸门触发；LP 对照未完整生成")
        bounds = formation_window_indices(reference_index, t)
        if bounds is None:
            continue
        lo, hi = bounds
        ym = pd.Period(t, freq="M")
        residuals = {key: window_residuals(R, X, lo, hi)
                     for key, (R, X) in matrices.items()}
        for variant, membership, industry, estimator, min_weeks in configs:
            legs = bases[membership].get(ym)
            if not legs:
                continue
            Res = residuals[industry]
            for f in FACTORS:
                sub = legs.get(f)
                if sub is None or sub.empty:
                    continue
                vals = {}
                for side, grp in [("long", N_GROUPS - 1), ("short", 0)]:
                    codes = sub.loc[sub["grp"] == grp, "code"].tolist()
                    fn = leave_one_out_correlations if estimator == "loo" else pairwise_correlations
                    vals[side] = fn(Res, codes, min_stock_weeks=min_weeks)
                if not (np.isfinite(vals["long"]["comomentum"])
                        and np.isfinite(vals["short"]["comomentum"])):
                    continue
                row = {"ym": str(ym), "factor": f, "variant": variant,
                       "comomentum": 0.5 * (vals["long"]["comomentum"] + vals["short"]["comomentum"])}
                for side in ["long", "short"]:
                    for key in ["n", "mean_weeks", "mean_pair_weeks"]:
                        row[f"{key}_{side}"] = vals[side][key]
                rows.append(row)
        del residuals
        log.info("LP comomentum 完成 %s", ym)

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("LP 对照没有任何有效测度")
    values = ["comomentum", "n_long", "n_short", "mean_weeks_long", "mean_weeks_short",
              "mean_pair_weeks_long", "mean_pair_weeks_short"]
    wide = long.pivot(index=["ym", "factor"], columns="variant", values=values)
    wide.columns = [f"comomentum_{v}" if a == "comomentum" else f"{a}_{v}"
                    for a, v in wide.columns]
    wide = wide.reset_index().sort_values(["ym", "factor"])
    write_parquet(wide, CLEAN / "comomentum_faithful.parquet")
    diag = diagnostics(wide)
    log.info("LP 对照测度：%s 行\n%s", len(wide), diag.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
