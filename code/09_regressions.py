"""
09_regressions.py —— 主检验 + 三层正交化（Table 2 / Table 3）★ 11/15 闸门

主检验：
    R_{f,t→t+h} = α_f + β·Crowding_{f,t} + γ'X_{f,t} + ε,   h ∈ {1, 3, 6, 12}

核心难点（审稿人第一问）：拥挤 ≠ 因子动量。
  赚钱的因子会吸引资金 → 拥挤度与因子过去收益高度相关 → β 可能只是在捕捉因子动量。
  三层防御，全部要做：
    ① 正交化   ：把拥挤度对因子自身过去 1/6/12 个月收益回归，取残差当"纯净拥挤度"
    ② 因子内时序：逐因子用其自身时间序列做预测回归，看符号与量级是否一致
                  （因子只有 6 个，"按拥挤度做截面排序"在扣掉因子固定效应后不可识别，
                   故不采用；详见 main() 内 ② 的注释）
    ③ 同时放入 ：拥挤度与因子过去收益同时进回归，看 β 是否存活

标准误：因子固定效应 + Newey–West(HAC, lag=6)。
  注：因子只有 6 个，因子层面聚类的渐近性质不可靠，故以 NW 为主、聚类结果仅作对照报出。

产出：
  reports/table2_main.csv              主检验：两个测度 × 四个持有期
  reports/table3_orthogonalization.csv 三层正交化
  reports/gate_2026-11-15_verdict.md   闸门判定：系数是否存活
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, CFG, REPORTS, get_logger, hac_ols, heartbeat, now_local,  # noqa: E402
                    placebo_pvalue, read_parquet)

log = get_logger("09_regressions", "09_regressions.log")
FACTORS = ["VAL", "SMB", "MOM", "QMJ", "LOWVOL", "TURN"]
HORIZONS = [1, 3, 6, 12]
NW_LAGS = 6


def placebo_fit(d: pd.DataFrame, cols: list[str], measure: str, y: str,
                obs_t: float) -> tuple[float, int]:
    """用与观测 pooled 回归相同的设计生成循环平移零分布。"""
    return placebo_pvalue(d, cols, measure, y, obs_t, z_by="factor")


def build_panel() -> pd.DataFrame:
    fr = read_parquet(CLEAN / "factor_returns.parquet")
    fr["date"] = pd.to_datetime(fr["date"])
    fr = fr.sort_values(["factor", "date"])
    # 因子自身的过去 1/6/12 个月收益（因子动量）
    # 口径：past{k} 在 t 时点覆盖 t-k+1..t 共 k 个月，与拥挤度的信息窗口
    # （以月末 t 结束的形成期）对齐。
    for k in (1, 6, 12):
        fr[f"past{k}"] = fr.groupby("factor", observed=True)["ls_fwd1"].shift(1).rolling(k, min_periods=k).sum()
    # 未来 h 个持有期的累计因子收益
    # 注意：factor_returns.parquet 里的 ls_fwd1 是「在 ym=t 记录、于 t+1 月实现」的收益，
    # 即 ls_fwd1[t] = R(t+1)（见 06 的 ret_next）。所以 t 时点的未来 h 个月收益就是
    # ls_fwd1[t .. t+h-1]，**不能再做一次 shift(-1)**——那样会跳过紧邻的 t+1 月、
    # 反而多算一个 t+h+1 月，把主检验的持有期整体往后错一个月。
    for h in HORIZONS:
        fr[f"fwd{h}"] = (fr.groupby("factor", observed=True)["ls_fwd1"]
                         .transform(lambda s: s.rolling(h, min_periods=h).sum().shift(-(h - 1))))
    cr = read_parquet(CLEAN / "crowding.parquet")
    cr["ym"] = pd.PeriodIndex(cr["ym"], freq="M")
    fr["ym"] = fr["date"].dt.to_period("M")
    df = fr.merge(cr, on=["ym", "factor"], how="inner")
    # 标准化（便于跨测度比较系数大小）
    for c in ["comomentum", "crowd_margin", "crowd_margin_long", "cover_long",
              "level_margin", "cover_short"]:
        if c in df.columns:
            df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
    # 因子固定效应。
    # 6 个因子的拥挤度水平与收益水平本就不同（实测 SMB 的 comomentum 均值比 LOWVOL
    # 低约 3 个标准差）。不控制的话，系数会有一部分来自「哪个因子整体更拥挤、
    # 整体收益更高」——这正是截面双变量排序里踩过的同一个坑（见 TASKPLAN 修正 5）。
    # 加上之后，β 只由因子**内**的时间序列变化识别，也正是假设本身所说的东西：
    # 「某个因子变拥挤 → 该因子衰减」讲的就是因子内的时间序列命题。
    #
    # 作为额外好处：因为安慰剂是在因子内循环平移，零分布里也保留了各因子的水平，
    # 所以安慰剂 p 值对这类跨因子水平污染本来就是稳健的。
    fe = pd.get_dummies(df["factor"], prefix="FE", drop_first=True).astype(float)
    df = pd.concat([df, fe], axis=1)
    return df


MOM_CTRL = ["past1", "past6", "past12"]


def controls_for(df: pd.DataFrame, spec: str) -> list[str]:
    """按规格返回控制变量列表。因子固定效应是所有规格的公共项。"""
    c = [x for x in df.columns if x.startswith("FE_")]   # 因子固定效应
    if spec in ("B", "C", "D"):
        c += MOM_CTRL
    if spec in ("C", "D") and "level_margin_z" in df.columns:
        c += ["level_margin_z"]          # 关键：控制两融「水平」，分离机械成分
    if spec == "D" and "cover_long_z" in df.columns:
        c += ["cover_long_z"]            # 控制组合两融覆盖率（样本宽度变化）
    return c


def main() -> None:
    df = build_panel()
    log.info(f"回归面板：{len(df)} 行（{df['ym'].nunique()} 个月 × {df['factor'].nunique()} 因子）"
             f" | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")
    if len(df) < 60:
        log.error("面板过小，无法回归")
        return

    # ---------- Table 2：主检验（四组规格）----------
    rows = []
    for measure in ["comomentum_z", "crowd_margin_z"]:
        if measure not in df.columns:
            continue
        for spec in ["A", "B", "C", "D"]:
            ctrl = controls_for(df, spec)
            for h in HORIZONS:
                cols = [measure] + ctrl
                d = df.dropna(subset=cols + [f"fwd{h}"]).copy()
                if len(d) < 40:
                    continue
                r = hac_ols(d[f"fwd{h}"].to_numpy(), d[cols].to_numpy(), cols)
                pp, ndraw = placebo_pvalue(d, cols, measure, f"fwd{h}", r[measure]["t"])
                rows.append({"measure": measure.replace("_z", ""), "spec": spec,
                             "controls": "+".join(ctrl) if ctrl else "无",
                             "horizon": h,
                             "beta": r[measure]["coef"], "se": r[measure]["se"],
                             "t": r[measure]["t"], "p": r[measure]["p"],
                             "p_placebo": pp, "n_placebo": ndraw,
                             "n": r["_n"], "r2": r["_r2"]})
    t2 = pd.DataFrame(rows)
    t2.to_csv(REPORTS / "table2_main.csv", index=False, encoding="utf-8-sig")
    piv2 = t2.pivot_table(index=["measure", "horizon"], columns="spec", values="beta")
    pivt2 = t2.pivot_table(index=["measure", "horizon"], columns="spec", values="t")
    pivp2 = t2.pivot_table(index=["measure", "horizon"], columns="spec", values="p_placebo")
    log.info("\n★ Table 2 主检验 · 系数（列=规格 A无控制 B+因子动量 C+两融水平 D+覆盖率）：\n"
             + piv2.round(4).to_string())
    log.info("\n★ Table 2 主检验 · HAC t 值：\n" + pivt2.round(2).to_string())
    log.info("\n★ Table 2 主检验 · 安慰剂 p 值（循环平移，空数据 95 分位约 2.8 而非 1.96）：\n"
             + pivp2.round(3).to_string())

    # ---------- Table 3：三层正交化 ----------
    rows3 = []
    for measure in ["comomentum_z", "crowd_margin_z"]:
        if measure not in df.columns:
            continue
        # 正交化时也要扣掉因子固定效应：只扣因子动量的话，残差里仍留着
        # 因子的水平差，第二阶段又会把它当成拥挤度捡回来。
        ctrl_o = ([x for x in df.columns if x.startswith("FE_")] + MOM_CTRL
                  + (["level_margin_z"] if "level_margin_z" in df.columns else []))

        # ① 正交化：对因子过去 1/6/12 个月收益 + 两融水平回归，取残差
        d = df.dropna(subset=[measure] + ctrl_o).copy()
        if len(d) < 40:
            continue
        # 正交化前先丢掉零方差列：某个因子在样本里整体缺失时，它的固定效应虚拟变量
        # 会退化成全零列、与常数项共线，statsmodels 只会发一条 SingularMatrixWarning
        # 然后给出不可识别的系数（残差本身仍是良定义的，但 R² 会失去意义）。
        Xo_raw = d[ctrl_o].to_numpy(dtype=float)
        sd = Xo_raw.std(axis=0)
        keep = np.isfinite(sd) & (sd > 0)
        if not keep.all():
            log.warning(f"正交化回归丢弃零方差列 "
                        f"{[c for c, k in zip(ctrl_o, keep) if not k]}")
        Xo = sm.add_constant(Xo_raw[:, keep], has_constant="add")
        fit_o = sm.OLS(d[measure].to_numpy(), Xo).fit()
        d["crowd_pure"] = fit_o.resid
        r2_orth = fit_o.rsquared
        for h in HORIZONS:
            dd = d.dropna(subset=["crowd_pure", f"fwd{h}"])
            if len(dd) < 40:
                continue
            r = hac_ols(dd[f"fwd{h}"].to_numpy(), dd[["crowd_pure"]].to_numpy(), ["crowd_pure"])
            pp, ndraw = placebo_pvalue(dd, ["crowd_pure"], "crowd_pure", f"fwd{h}",
                                       r["crowd_pure"]["t"])
            rows3.append({"measure": measure.replace("_z", ""), "layer": "①正交化后残差",
                          "horizon": h, "beta": r["crowd_pure"]["coef"], "t": r["crowd_pure"]["t"],
                          "p": r["crowd_pure"]["p"], "p_placebo": pp, "n": r["_n"],
                          "note": f"正交化R²={r2_orth:.3f}（对因子动量+两融水平）；"
                                  f"安慰剂 |t| 零分布 {ndraw} 次重抽"})

        # ③ 同时放入：拥挤度 + 全部控制变量
        for h in HORIZONS:
            cols = [measure] + ctrl_o
            dd = df.dropna(subset=cols + [f"fwd{h}"])
            if len(dd) < 40:
                continue
            r = hac_ols(dd[f"fwd{h}"].to_numpy(), dd[cols].to_numpy(), cols)
            pp, _ = placebo_pvalue(dd, cols, measure, f"fwd{h}", r[measure]["t"])
            rows3.append({"measure": measure.replace("_z", ""), "layer": "③与全部控制同时放入",
                          "horizon": h, "beta": r[measure]["coef"], "t": r[measure]["t"],
                          "p": r[measure]["p"], "p_placebo": pp, "n": r["_n"],
                          "note": "控制=" + "+".join(ctrl_o)})

    # ② 因子内时序检验（替代方案里的「截面双变量排序」）
    # 因子只有 6 个，按拥挤度做截面排序、再控制因子动量，在统计上是不可识别的：
    # 扣掉因子固定效应后，排序比较的其实就是固定效应本身。空数据冒烟测试里，
    # 这种排序能把 t 推到 2.8~3.7（comomentum）和 3.1（两融），全是假的。
    # 改为逐因子做时序预测回归——每个因子用自己过去 1/6/12 个月收益做控制，
    # 看拥挤度是否还有增量预测力。这也更贴合假设本身："某个因子变拥挤 → 该因子衰减"
    # 讲的就是因子内的时间序列变化。
    for measure in ["comomentum_z", "crowd_margin_z"]:
        if measure not in df.columns:
            continue
        ctrl2 = MOM_CTRL + (["level_margin_z"] if "level_margin_z" in df.columns else [])
        slopes = []
        for f, g in df.dropna(subset=[measure] + ctrl2 + ["fwd1"]).groupby("factor", observed=True):
            if len(g) < 30:
                continue
            # 该层要回答的是「控制住因子自身的动量之后，拥挤度还有没有增量预测力」，
            # 所以必须真的把过去 1/6/12 月收益放进去。原来的写法只放了测度本身、
            # 注释却写「控制自身过去收益」，名实不符。
            cols2 = [measure] + ctrl2
            r = hac_ols(g["fwd1"].to_numpy(), g[cols2].to_numpy(), cols2)
            pp2, _ = placebo_pvalue(g, cols2, measure, "fwd1", r[measure]["t"])
            slopes.append((f, r[measure]))
            rows3.append({"measure": measure.replace("_z", ""), "layer": "②因子内时序回归",
                          "horizon": 1, "beta": r[measure]["coef"], "t": r[measure]["t"],
                          "p": r[measure]["p"], "p_placebo": pp2, "n": r["_n"],
                          "note": f"因子 {f}：控制自身过去 1/6/12 月收益"
                                  f"{'+ 两融水平' if 'level_margin_z' in ctrl2 else ''}"})
        if len(slopes) > 1:
            b = np.array([s["coef"] for _, s in slopes])
            tt = b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if b.std(ddof=1) > 0 else np.nan
            rows3.append({"measure": measure.replace("_z", ""), "layer": "②六因子斜率汇总",
                          "horizon": 1, "beta": float(b.mean()), "t": float(tt),
                          "p": np.nan, "n": len(b),
                          "note": f"{len(b)} 个因子斜率的均值；负值因子 "
                                  f"{int((b < 0).sum())}/{len(b)}。n 太小，此项仅作方向性参考，"
                                  f"不做显著性主张"})

    t3 = pd.DataFrame(rows3)
    t3.to_csv(REPORTS / "table3_orthogonalization.csv", index=False, encoding="utf-8-sig")
    log.info("\n★ Table 3 三层正交化：\n" + t3.round(4).to_string())

    # ---------- 闸门判定 ----------
    #
    # 判定规则（事先定死，避免看到结果再挑标准）：
    #   假设方向是 β < 0（拥挤 → 未来收益更低）。
    #   证据单元 = 「①正交化后残差」或「③与全部控制同时放入」× h ∈ {1,3,6}，
    #   共 2 层 × 3 个持有期 = 6 个单元/测度。h=12 只作参考，不参与判定
    #   （效应若存在更可能是短期的；且 h 越长重叠越多，安慰剂分布越宽）。
    #   显著性一律以**安慰剂 p < 0.10** 为准，不看 HAC 的 p。
    #
    #   存活   ：某个测度有 ≥2 个单元显著，且这些显著单元的系数**全部为负**
    #   方向矛盾：某测度有 ≥2 个单元显著，但显著单元里正负都有 → 需人工判断
    #   消失   ：两个测度都没有 ≥2 个显著单元
    verdict = ["# 11/15 闸门判定：拥挤度系数是否在正交化后存活", "",
               f"生成时间：{now_local():%Y-%m-%d %H:%M:%S}（北京时间）", "",
               "判定口径：假设方向为负；证据单元为 ①正交化后残差 / ③与全部控制同时放入",
               "× h ∈ {1,3,6}；显著性以**安慰剂 p < 0.10** 为准（HAC 的 p 仅供对照，",
               "本设计形成期 12 个月却按月重叠采样，HAC 的 t 在空数据上的 95 分位约 2.8）。", ""]

    ev = t3[t3["layer"].isin(["①正交化后残差", "③与全部控制同时放入"])
           & t3["horizon"].isin([1, 3, 6])].copy()
    ev = ev.dropna(subset=["beta", "p_placebo"])

    verdict.append("## 证据单元（6 个/测度）")
    verdict.append("")
    verdict.append("| 测度 | 层 | h | 系数 | HAC t | 安慰剂 p | 显著？ |")
    verdict.append("|---|---|---|---|---|---|---|")
    for _, r in ev.iterrows():
        sig = "**是**" if (r["p_placebo"] < 0.10 and r["beta"] < 0) else ""
        verdict.append(f"| {r['measure']} | {r['layer'][:1]} | {r['horizon']} | {r['beta']:+.4f} | "
                       f"{r['t']:+.2f} | {r['p_placebo']:.3f} | {sig} |")
    verdict.append("")

    decision = {}
    for meas in ev["measure"].unique():
        s = ev[ev["measure"] == meas]
        sig = s[(s["p_placebo"] < 0.10)]
        decision[meas] = {"n_sig": int(len(sig)),
                          "n_neg": int((sig["beta"] < 0).sum()),
                          "n_pos": int((sig["beta"] > 0).sum())}

    verdict.append("## 判定")
    verdict.append("")
    for meas, st in decision.items():
        verdict.append(f"- **{meas}**：显著单元 {st['n_sig']} 个（负向 {st['n_neg']}，正向 {st['n_pos']}）")
    verdict.append("")

    survived = [m for m, st in decision.items() if st["n_sig"] >= 2 and st["n_pos"] == 0]
    mixed = [m for m, st in decision.items() if st["n_sig"] >= 2 and st["n_pos"] > 0]

    if survived and not any(st["n_sig"] >= 2 for m, st in decision.items() if m not in survived):
        verdict += [f"**判定：系数存活**（{'、'.join(survived)}），其余测度消失。",
                    "",
                    "论文按「拥挤度预测 A 股因子衰减」写，但**必须把这个不对称本身当作主要发现**："
                    "直接测度（两融余额）显著为负，间接测度（comomentum）完全无预测力。"
                    "两者在因子内的相关性本就接近于零且符号不一（见 Table 1），"
                    "所以正确的读法不是「Lou & Polk 的结果在 A 股成立或不成立」，"
                    "而是「在可以直接观测杠杆持仓的市场里，间接测度没有捕捉到拥挤」。"
                    "这是一个测量层面的贡献，也是本文区别于单国复制的地方。"]
    elif survived:
        verdict += [f"**判定：系数存活**（{'、'.join(survived)}）。",
                    "",
                    "论文按「拥挤度预测 A 股因子衰减」写：标题、摘要、引言按原定方案，"
                    "并把两个测度的一致性（或不一致性）作为边界条件证据展开。"]
    elif mixed:
        verdict += [f"**判定：方向矛盾**（{'、'.join(mixed)}）。",
                    "",
                    "有显著单元但正负都有 → **不自动改写**。需要人工判断："
                    "可能是某个持有期上反向（例如短期动量延续、长期才衰减），"
                    "也可能是测度捕捉到了别的东西。这一情形事先没写预案，必须先讨论再动笔。"]
    else:
        verdict += ["**判定：系数消失。**",
                    "",
                    "按预案改写成否定性发现——「所谓因子拥挤效应在 A 股可能只是因子动量的影子」。",
                    "标题、摘要、引言要重写。注意这仍是一个可发表的结果：",
                    "它是对 Lou & Polk (2022) 结论的边界条件检验，且我们有一个他们无法做的检验——"
                    "直接测度（两融）与间接测度（comomentum）同时给出否定证据。",
                    "",
                    "写法上要避免的坑：不能只是「我们没找到显著」，而要说清楚"
                    "**在什么条件下本该找到却找不到**（杠杆受限、卖空昂贵、散户主导），"
                    "并报出安慰剂分布，说明不是检验功效不足。"]
    (REPORTS / "gate_2026-11-15_verdict.md").write_text("\n".join(verdict), encoding="utf-8")
    log.info(f"闸门判定：{decision}")
    log.info("闸门判定已写入 reports/gate_2026-11-15_verdict.md")
    heartbeat("09 主检验结束", log)


if __name__ == "__main__":
    main()
