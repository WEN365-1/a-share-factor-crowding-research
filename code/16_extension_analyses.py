"""
16_extension_analyses.py —— 投稿前补强分析（对应审稿前补强清单的 H4/H11/H12/M7 + 拓展）

产出（全部写入 reports/，英文标签，供 12/18 生成 LaTeX 表）：
  table6_compound.csv      H4：复利收益 vs 求和收益（spec D，两测度 × h∈{1,3,6,12}）
  table8_legs.csv          H12：腿分解（多空/多头/空头腿，测度=总拥挤度与各腿自身拥挤度）
  table9_robust_extra.csv  H11/M7：剔除 SMB、覆盖率门槛、within-factor 标准化
  table10_agreement.csv    拓展：两代理一致性拆分（comomentum 高/低时 margin 系数的差异）
  table_cov_stats.csv      H11a：各因子两融覆盖率统计（用于正文披露）

设计口径与 09 完全一致：pooled 标准化、因子固定效应、spec D 控制变量；
h=12 用 Newey–West(12)，其余期限用 Newey–West(6)，观测与置换同阶数；
因子内循环平移安慰剂 p 值采用加一修正。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, REPORTS, get_logger, hac_ols, heartbeat,  # noqa: E402
                    placebo_pvalue, placebo_null_dist, read_parquet, time_index)

log = get_logger("16_ext", "16_extension_analyses.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
HORIZONS = [1, 3, 6, 12]
MOM_CTRL = ["past1", "past6", "past12"]


# ------------------------------------------------------------------
# 面板构造（镜像 09.build_panel，另加腿级与复利收益）
# ------------------------------------------------------------------
def build_panel() -> pd.DataFrame:
    fr = read_parquet(CLEAN / "factor_returns.parquet")
    fr["date"] = pd.to_datetime(fr["date"])
    fr = fr.sort_values(["factor", "date"])
    for k in (1, 6, 12):
        fr[f"past{k}"] = (fr.groupby("factor", observed=True)["ls_fwd1"]
                          .shift(1).rolling(k, min_periods=k).sum())
    for h in HORIZONS:
        fr[f"fwd{h}"] = (fr.groupby("factor", observed=True)["ls_fwd1"]
                         .transform(lambda s: s.rolling(h, min_periods=h).sum().shift(-(h - 1))))
        # 复利（几何累乘）：fwd{h}_c = prod(1+r_{t+1..t+h}) - 1
        fr[f"fwd{h}_c"] = (fr.groupby("factor", observed=True)["ls_fwd1"]
                           .transform(lambda s: (1.0 + s).rolling(h, min_periods=h)
                                      .apply(np.prod, raw=True).shift(-(h - 1)) - 1.0))
        # 腿级持有期收益：top_ret / bot_ret 在 t 行记录的都是 t+1 月实现的收益
        for leg in ("top", "bot"):
            fr[f"fwd{h}_{leg}"] = (fr.groupby("factor", observed=True)[f"{leg}_ret"]
                                   .transform(lambda s: s.rolling(h, min_periods=h)
                                              .sum().shift(-(h - 1))))
        # 腿级自身动量
    for k in (1, 6, 12):
        for leg in ("top", "bot"):
            fr[f"past{k}_{leg}"] = (fr.groupby("factor", observed=True)[f"{leg}_ret"]
                                    .shift(1).rolling(k, min_periods=k).sum())

    cr = read_parquet(CLEAN / "crowding.parquet")
    cr["ym"] = pd.PeriodIndex(cr["ym"], freq="M")
    fr["ym"] = fr["date"].dt.to_period("M")
    df = fr.merge(cr, on=["ym", "factor"], how="inner")
    # 空头腿拥挤度：crowd_margin = v_long - v_short => v_short = v_long - crowd_margin
    df["crowd_margin_short"] = df["crowd_margin_long"] - df["crowd_margin"]
    for c in ["comomentum", "crowd_margin", "crowd_margin_long", "crowd_margin_short",
              "cover_long", "cover_short", "level_margin"]:
        if c in df.columns:
            df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
    fe = pd.get_dummies(df["factor"], prefix="FE", drop_first=True).astype(float)
    df = pd.concat([df, fe], axis=1)
    return df.sort_values(["ym", "factor"]).copy()


def controls_for(df: pd.DataFrame, spec: str) -> list[str]:
    c = [x for x in df.columns if x.startswith("FE_")]
    if spec in ("B", "C", "D"):
        c += MOM_CTRL
    if spec in ("C", "D") and "level_margin_z" in df.columns:
        c += ["level_margin_z"]
    if spec == "D" and "cover_long_z" in df.columns:
        c += ["cover_long_z", "cover_short_z"]
    return c


def fit_row(d: pd.DataFrame, cols: list[str], measure: str, y: str,
            meta: dict) -> dict | None:
    """HAC + 循环平移安慰剂，与 09 完全一致的推断。"""
    d = d.dropna(subset=cols + [y]).sort_values(["ym", "factor"]).copy()
    if len(d) < 40:
        return None
    lags = 12 if meta.get("horizon") == 12 else 6
    # A 类：因子 × 月堆叠面板 → Driscoll–Kraay（滞后按月度而非行）。
    r = hac_ols(d[y].to_numpy(), d[cols].to_numpy(), cols, lags=lags,
                time=time_index(d["ym"]))
    pp, nd = placebo_pvalue(d, cols, measure, y, r[measure]["t"], hac_lags=lags)
    row = {"measure": measure.replace("_z", ""), "horizon": None, "hac_lags": lags,
           "beta": r[measure]["coef"], "se": r[measure]["se"], "t": r[measure]["t"],
           "p": r[measure]["p"], "p_placebo": pp, "n": r["_n"]}
    row.update(meta)
    return row


# ------------------------------------------------------------------
# 1) H4：复利 vs 求和收益
# ------------------------------------------------------------------
def compound_returns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ctrl = controls_for(df, "D")
    for measure in ["comomentum_z", "crowd_margin_z"]:
        for h in HORIZONS:
            for ykind, yname in (("sum", f"fwd{h}"), ("compound", f"fwd{h}_c")):
                r = fit_row(df, [measure] + ctrl, measure, yname,
                            {"analysis": "return definition", "group": ykind,
                             "horizon": h})
                if r:
                    rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "table6_compound.csv", index=False, encoding="utf-8-sig")
    log.info(f"table6_compound.csv：{len(out)} 行")
    return out


# ------------------------------------------------------------------
# 2) H12：腿分解
# ------------------------------------------------------------------
def leg_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ctrlD = controls_for(df, "D")
    # 2a) 总拥挤度 → 各腿（RHS 与主检验 D 完全一致 → β_LS = β_top − β_bot 精确成立）
    for measure in ["comomentum_z", "crowd_margin_z"]:
        for h in [1, 6]:
            cols = [measure] + ctrlD
            base = df.dropna(subset=cols + [f"fwd{h}", f"fwd{h}_top", f"fwd{h}_bot"]).copy()
            for leg, y in (("long-short", f"fwd{h}"), ("long", f"fwd{h}_top"),
                           ("short", f"fwd{h}_bot")):
                r = fit_row(base, cols, measure, y,
                            {"analysis": "leg decomposition (same RHS)",
                             "group": leg, "horizon": h})
                if r:
                    rows.append(r)
    # 2b) 各腿自身拥挤度 → 各腿自身收益（控制腿自身动量 + 对应覆盖率）
    for h in [1, 6]:
        # 多头：measure=crowd_margin_long_z, y=fwd{h}_top, ctrl=FE+past_top+level+cover_long
        ctrl_top = ([c for c in df.columns if c.startswith("FE_")]
                    + ["past1_top", "past6_top", "past12_top", "level_margin_z", "cover_long_z"])
        r = fit_row(df, ["crowd_margin_long_z"] + ctrl_top, "crowd_margin_long_z",
                    f"fwd{h}_top", {"analysis": "own-leg measure", "group": "long",
                                    "horizon": h})
        if r:
            rows.append(r)
        # 空头：measure=crowd_margin_short_z, y=fwd{h}_bot, ctrl=FE+past_bot+level+cover_short
        ctrl_bot = ([c for c in df.columns if c.startswith("FE_")]
                    + ["past1_bot", "past6_bot", "past12_bot", "level_margin_z", "cover_short_z"])
        r = fit_row(df, ["crowd_margin_short_z"] + ctrl_bot, "crowd_margin_short_z",
                    f"fwd{h}_bot", {"analysis": "own-leg measure", "group": "short",
                                    "horizon": h})
        if r:
            rows.append(r)
        # 对称：多头拥挤度 → 空头腿、空头拥挤度 → 多头腿（交叉腿，判断溢出）
        r = fit_row(df, ["crowd_margin_long_z"] + ctrl_bot, "crowd_margin_long_z",
                    f"fwd{h}_bot", {"analysis": "cross-leg measure", "group": "long->short",
                                    "horizon": h})
        if r:
            rows.append(r)
        r = fit_row(df, ["crowd_margin_short_z"] + ctrl_top, "crowd_margin_short_z",
                    f"fwd{h}_top", {"analysis": "cross-leg measure", "group": "short->long",
                                    "horizon": h})
        if r:
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "table8_legs.csv", index=False, encoding="utf-8-sig")
    log.info(f"table8_legs.csv：{len(out)} 行")
    return out


# ------------------------------------------------------------------
# 3) H11/M7：剔除 SMB、覆盖率门槛、within-factor 标准化
# ------------------------------------------------------------------
def robust_extra(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ctrl = controls_for(df, "D")
    specs = [
        ("exclude SMB", df[df["factor"] != "SMB"].copy(), "crowd_margin_z"),
        ("exclude SMB", df[df["factor"] != "SMB"].copy(), "comomentum_z"),
        ("cover>=0.30", df[df["cover_long"] >= 0.30].copy(), "crowd_margin_z"),
        ("cover>=0.30", df[df["cover_long"] >= 0.30].copy(), "comomentum_z"),
        ("cover>=0.50", df[df["cover_long"] >= 0.50].copy(), "crowd_margin_z"),
        ("cover>=0.50", df[df["cover_long"] >= 0.50].copy(), "comomentum_z"),
    ]
    for name, sub, measure in specs:
        for h in HORIZONS:
            r = fit_row(sub, [measure] + ctrl, measure, f"fwd{h}",
                        {"analysis": name, "group": name, "horizon": h})
            if r:
                rows.append(r)

    # within-factor 标准化：组内去均值后除以组内（合并）标准差
    dfw = df.copy()
    for base in ["comomentum", "crowd_margin"]:
        dm = dfw[base] - dfw.groupby("factor", observed=True)[base].transform("mean")
        dfw[base + "_w"] = dm / dm.std()
    for measure in ["comomentum_w", "crowd_margin_w"]:
        for h in HORIZONS:
            r = fit_row(dfw, [measure] + ctrl, measure, f"fwd{h}",
                        {"analysis": "within-factor standardization", "group": "within",
                         "horizon": h})
            if r:
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "table9_robust_extra.csv", index=False, encoding="utf-8-sig")
    log.info(f"table9_robust_extra.csv：{len(out)} 行")
    return out


# ------------------------------------------------------------------
# 4) 拓展：两代理一致性拆分（comomentum 高/低时 margin 系数的差异）
# ------------------------------------------------------------------
def placebo_null_interaction(d: pd.DataFrame, cols: list[str], base: str,
                             inter: str, y: str, hac_lags: int = 6) -> np.ndarray:
    """交互项的循环平移零分布：平移 base 测度后**重建**交互列再重估。"""
    from common import _grouped_circular_shift
    frame = d.sort_values(["ym", "factor"]).copy()
    groups = [g for _, g in frame.groupby("factor", observed=True, sort=False)]
    if not groups or any(len(g) < 30 for g in groups):
        return np.asarray([], dtype=float)
    n_shifts = min(len(g) for g in groups) - 1
    tvals = time_index(frame["ym"])   # A 类堆叠面板 → DK
    ts = []
    for k in range(1, n_shifts + 1):
        perm = _grouped_circular_shift(frame, base, "factor", k)
        perm[inter] = perm[base] * perm["high_comom"]
        r = hac_ols(perm[y].to_numpy(dtype=float), perm[cols].to_numpy(dtype=float), cols,
                    lags=hac_lags, time=tvals)
        t = r.get(inter, {}).get("t", np.nan)
        if np.isfinite(t):
            ts.append(abs(float(t)))
    return np.asarray(ts, dtype=float)


def agreement_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ctrl = controls_for(df, "D")
    # 组内中位数拆分：comomentum 高于该因子自身中位数的月份
    med = df.groupby("factor", observed=True)["comomentum"].transform("median")
    df["high_comom"] = (df["comomentum"] > med).astype(float)
    for name, sub in (("high comomentum", df[df["high_comom"] == 1].copy()),
                      ("low comomentum", df[df["high_comom"] == 0].copy())):
        for h in HORIZONS:
            r = fit_row(sub, ["crowd_margin_z"] + ctrl, "crowd_margin_z", f"fwd{h}",
                        {"analysis": "agreement split", "group": name, "horizon": h})
            if r:
                rows.append(r)
    # 交互项：crowd_margin_z × high_comom（平移后重建交互列的零分布）
    for h in HORIZONS:
        lags = 12 if h == 12 else 6
        cols = ["crowd_margin_z", "high_comom", "inter_mh"] + ctrl
        d = df.copy()
        d["inter_mh"] = d["crowd_margin_z"] * d["high_comom"]
        d = d.dropna(subset=cols + [f"fwd{h}"]).copy()
        if len(d) < 40:
            continue
        r = hac_ols(d[f"fwd{h}"].to_numpy(), d[cols].to_numpy(), cols, lags=lags,
                    time=time_index(d["ym"]))
        null = placebo_null_interaction(d, cols, "crowd_margin_z", "inter_mh", f"fwd{h}",
                                        hac_lags=lags)
        pp = (float((1.0 + np.sum(null >= abs(r["inter_mh"]["t"]))) / (1.0 + null.size))
              if null.size else np.nan)
        rows.append({"measure": "interaction", "analysis": "agreement split",
                     "group": "margin x high comomentum", "horizon": h, "hac_lags": lags,
                     "beta": r["inter_mh"]["coef"], "se": r["inter_mh"]["se"],
                     "t": r["inter_mh"]["t"], "p": r["inter_mh"]["p"],
                     "p_placebo": pp, "n": r["_n"]})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "table10_agreement.csv", index=False, encoding="utf-8-sig")
    log.info(f"table10_agreement.csv：{len(out)} 行")
    return out


# ------------------------------------------------------------------
# 5) H11a：覆盖率统计（正文披露用）
# ------------------------------------------------------------------
def coverage_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FACTORS:
        g = df[df["factor"] == f]
        rows.append({
            "factor": f, "n_months": int(len(g)),
            "cover_long_mean": float(g["cover_long"].mean()),
            "cover_long_min": float(g["cover_long"].min()),
            "cover_long_p10": float(g["cover_long"].quantile(0.10)),
            "cover_short_mean": float(g["cover_short"].mean()),
            "share_below_030": float((g["cover_long"] < 0.30).mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "table_cov_stats.csv", index=False, encoding="utf-8-sig")
    log.info("table_cov_stats.csv：\n" + out.round(3).to_string())
    return out


def main() -> None:
    df = build_panel()
    log.info(f"扩展分析面板：{len(df)} 行（{df['ym'].nunique()} 个月 × {df['factor'].nunique()} 因子）")
    compound_returns(df)
    leg_decomposition(df)
    robust_extra(df)
    agreement_split(df)
    coverage_stats(df)
    # 主规范经济量级（h=6, D）：pooled 与 within 两种标准化的对照
    ctrl = controls_for(df, "D")
    for base in ["crowd_margin", "comomentum"]:
        d = df.dropna(subset=[base + "_z"] + ctrl + ["fwd6"])
        r = hac_ols(d["fwd6"].to_numpy(), d[[base + "_z"] + ctrl].to_numpy(),
                    [base + "_z"] + ctrl, time=time_index(d["ym"]))
        sd_w = float((d[base] - d.groupby("factor", observed=True)[base]
                      .transform("mean")).std())
        log.info(f"{base}: h=6 D β_z={r[base+'_z']['coef']:+.4f} "
                 f"(t={r[base+'_z']['t']:+.2f}) | pooled SD={d[base].std():.4f} "
                 f"within SD={sd_w:.4f} | +1SD 效应 = {r[base+'_z']['coef']*100:+.2f}pp(池化口径)")
    heartbeat("16 扩展分析结束", log)


if __name__ == "__main__":
    main()
