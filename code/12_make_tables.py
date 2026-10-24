"""
12_make_tables.py —— 把 reports/*.csv 转成论文用的 LaTeX 表，并导出图片为 PDF

为什么单独一步、不直接在分析脚本里拼 LaTeX：
  分析脚本要能在没有 LaTeX 环境下重跑，而且表格的**排版口径**（有效数字、
  星号依据、变量名转义）会反复调整——混在一起会让每次改格式都要重跑半小时的回归。

星号口径：**一律以安慰剂 p 值为准**，不用 HAC 的 p。
  本设计形成期 12 个月却按月重叠采样，空数据上 HAC 的 |t| 95 分位约 2.8 而非 1.96，
  用 HAC 的 p 打星号会系统性高估显著性（空数据上 32 个规格里有 5 个会被误标）。

产出（写入 paper/tables/ 与 paper/figures/）：
  table1_corr.tex   两个拥挤度测度的水平与相关性
  table2_main.tex   主检验：两个测度 × 四个规格 × 四个持有期
  table3_orth.tex   三层正交化
  table4_tail.tex   尾部特征：高/低拥挤期的对比
  table5_robust.tex 稳健性
  figures/fig_crowding_series.pdf
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, get_logger, heartbeat, now_local  # noqa: E402

log = get_logger("12_tables", "12_make_tables.log")
PAPER = Path(__file__).resolve().parent.parent / "paper"
TABDIR = PAPER / "tables"
FIGDIR = PAPER / "figures"

SPEC_ZH = {"A": "A", "B": "B", "C": "C", "D": "D"}

# 分析脚本里为了可读性用了中文标签，但**表内绝不能出现中文**（投稿硬要求，也是自己的检查清单第一条）。
# 在 LaTeX 生成层做映射，既保持 CSV 的中文可读性，又不让中文进论文。
EN = {
    # 测度名（分析脚本里用下划线，论文里要有空格）
    "comomentum": "Comomentum",
    "crowd_margin": "Margin-balance proxy",
    "crowd_margin_long": "Margin-balance proxy, long leg",
    "crowd_flow_buy": "Financing purchases (buy flow)",
    "crowd_flow_net": "Net financing flows (SSE)",
    "leg decomposition (same RHS)": "Total measure, identical controls",
    "own-leg measure": "Own-leg measure, own-leg return",
    "cross-leg measure": "Cross-leg measure",
    "long-short": "Long-short", "long": "Long leg", "short": "Short leg",
    "long->short": "Long measure $\\rightarrow$ short leg".replace("measure $\\rightarrow$", "$\\rightarrow$"),
    "short->long": "Short measure $\\rightarrow$ long leg".replace("measure $\\rightarrow$", "$\\rightarrow$"),
    # 因子名
    "POOLED": "Pooled", "POOLED(因子内去均值)": "Pooled (factor-demeaned)",
    # Table 3 的层名
    "①正交化后残差": "Layer 1: orthogonalised residual",
    "③与全部控制同时放入": "Layer 3: all controls simultaneously",
    "②因子内时序回归": "Layer 2: within-factor time series",
    "②六因子斜率汇总": "Layer 2: cross-factor mean slope",
    # Table 5 的检验与分组
    "子样本": "Subsample", "等权组合": "Equal-weighted", "构造窗口": "Window",
    "测度变体": "Variant",
    "2015–2018 杠杆牛熊": "2015--2018 leverage cycle",
    "2019–2021 北向+主动基金共振": "2019--2021 foreign inflows",
    "2022–2025 两融散户主导": "2022--2025 retail margin",
    "等权（不按市值加权）": "Equal-weighted",
    "两融余额 6 个月变化": "Balance change, 6 months",
    "两融余额 24 个月变化": "Balance change, 24 months",
    "仅多头端（中国两融高度不对称）": "Long leg only",
    "仅空头端": "Short leg only",
    "crowd_margin_short": "Margin-balance proxy, short leg",
    # Table 4 的高低分组
    "高拥挤_": "High crowd.", "低拥挤_": "Low crowd.",
    "均值": "Mean", "标准差": "SD", "偏度": "Skew.", "峰度": "Kurt.", "崩溃频率": "Crash freq.",
}


def en(x) -> str:
    """把分析脚本里的中文标签换成论文用的英文；未知标签原样保留并转义。"""
    s = str(x)
    return EN.get(s, esc(s))


# ---------- 排版小工具 ----------

def esc(s: str) -> str:
    """转义 LaTeX 特殊字符（变量名里的下划线是常客）。"""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def notes_row(ncols: int, extra: str = "") -> str:
    """Notes 行：放在可换行的段落盒里（长文本不越界），统一星号口径说明。"""
    star = ("Stars use finite circular-shift permutation $p$-values; "
            "* $p<0.10$, ** $p<0.05$, *** $p<0.01$.")
    text = (extra + " " + star) if extra else star
    return (r"\multicolumn{" + str(ncols) + r"}{l}{\parbox{\dimexpr\textwidth-10pt\relax}"
            r"{\footnotesize Notes: " + text + r"}} \\")


def stars(p) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def num(x, nd: int = 4) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.{nd}f}"


def num_fixed(x, nd: int) -> str:
    """固定小数位，避免同一表列因来源字段不同而漂移。"""
    return num(x, nd)


def write(name: str, body: str, size: str = r"\small", sep: str = "2.5pt") -> None:
    TABDIR.mkdir(parents=True, exist_ok=True)
    # 表格按自然宽度排版（不用 resizebox 拉满行宽：小表会被吹大、大表被压小）；
    # 宽表用更小字号或更小列距。
    if name.startswith("table"):
        body = ("\\begingroup" + size + "\\setlength{\\tabcolsep}{" + sep + "}\n"
                + body + "\n\\endgroup")
    (TABDIR / name).write_text(body, encoding="utf-8")
    log.info(f"写出 paper/tables/{name}")


# ---------- Table 1：两个测度的相关性 ----------

def table1() -> None:
    p = REPORTS / "crowding_correlation.csv"
    if not p.exists():
        log.warning("缺 crowding_correlation.csv，跳过 Table 1")
        return
    d = pd.read_csv(p)
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Factor & $N$ & Pearson & Spearman & Mean comomentum & Mean margin-balance proxy \\",
             r"\midrule"]
    for _, r in d.iterrows():
        f = r["factor"]
        label = f"\\textit{{{en(f)}}}" if "POOLED" in str(f) else en(f)
        lines.append(f"{label} & {int(r['n_months'])} & {num(r['pearson'],3)} & "
                     f"{num(r['spearman'],3)} & {num(r['comomentum_mean'],5)} & "
                     f"{num(r['crowd_mean'],4)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table1_corr.tex", "\n".join(lines) + "\n", size=r"\footnotesize")


# ---------- Table 2：主检验 ----------

def table2() -> None:
    """主检验：单面板，每格系数（星号）+ 括号内置换 p 值。"""
    p = REPORTS / "table2_main.csv"
    if not p.exists():
        log.warning("缺 table2_main.csv，跳过 Table 2")
        return
    d = pd.read_csv(p)
    specs = ["A", "B", "C", "D"]
    horizons = sorted(d["horizon"].unique())
    out = [r"\begin{tabular}{ll" + "c" * len(specs) + "}", r"\toprule",
           r"Measure & $h$ & " + " & ".join(SPEC_ZH[s] for s in specs) + r" \\",
           r"\midrule"]
    for meas in ["comomentum", "crowd_margin"]:
        sub = d[d["measure"] == meas]
        for h in horizons:
            cells = []
            for spec in specs:
                c = sub[(sub["horizon"] == h) & (sub["spec"] == spec)]
                if c.empty:
                    cells.append(r"\shortstack{—\\(—)}")
                    continue
                c = c.iloc[0]
                cells.append(r"\shortstack{" + f"{c['beta']:,.4f}" + stars(c["p_placebo"])
                             + r"\\(" + f"{c['p_placebo']:.3f}" + r")}")
            out.append(f"{en(meas)} & {h} & " + " & ".join(cells) + r" \\")
    out += [r"\midrule",
            notes_row(6, extra=("Each cell reports the coefficient, starred by the finite "
                                "circular-shift permutation $p$-value, with the permutation "
                                "$p$-value in parentheses.")),
            r"\bottomrule", r"\end{tabular}"]
    write("table2_main.tex", "\n".join(out) + "\n")

    # 未直接放入短正文的推断附表也必须由同一份 CSV 重建，避免旧结果残留。
    for field, title, nd, name in [
        ("t", "HAC $t$-statistic", 2, "table2_t.tex"),
        ("p_placebo", "Finite permutation $p$-value", 3, "table2_p.tex"),
    ]:
        inference = [r"\begin{tabular}{llcccc}", r"\toprule",
                     rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}}}} \\",
                     r"Measure & $h$ & A & B & C & D \\", r"\midrule"]
        for meas in ["comomentum", "crowd_margin"]:
            sub = d[d["measure"] == meas]
            for h in horizons:
                cells = []
                for spec in specs:
                    c = sub[(sub["horizon"] == h) & (sub["spec"] == spec)]
                    cells.append("—" if c.empty else num_fixed(c.iloc[0][field], nd))
                inference.append(f"{en(meas)} & {h} & " + " & ".join(cells) + r" \\")
        inference += [r"\bottomrule", r"\end{tabular}"]
        write(name, "\n".join(inference) + "\n")


# ---------- Table 3：三层正交化 ----------

def table3() -> None:
    p = REPORTS / "table3_orthogonalization.csv"
    if not p.exists():
        log.warning("缺 table3_orthogonalization.csv，跳过 Table 3")
        return
    d = pd.read_csv(p)
    # Layer 2 的逐因子行只作方向性参考（正文已说明），不入表；仅保留均值汇总行，
    # 逐因子明细留在复制输出里，并在表注给出符号分布。
    l2 = d[d["layer"].str.contains("②因子内时序回归", na=False)]
    d = d[~d["layer"].str.contains("②因子内时序回归", na=False)]
    d = d[d["horizon"] != 12]   # h=12 为辅助口径，正文不引用，留复制输出
    l2_note = ""
    for meas in ["crowd_margin", "comomentum"]:
        g = l2[l2["measure"] == meas]
        if len(g):
            neg = int((g["beta"] < 0).sum())
            sig = int(((g["beta"] < 0) & (g["p_placebo"] < 0.10)).sum())
            l2_note += (f"{en(meas)} factor-specific slopes are negative for {neg}/6 "
                        f"factors ({sig} significant at the 10\\% level); ")
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r"Measure & Layer & $h$ & $\beta$ & HAC $t$ & Perm. $p$ \\", r"\midrule"]
    for _, r in d.iterrows():
        pv = r.get("p_placebo")
        out.append(f"{en(r['measure'])} & {en(r['layer'])} & {r['horizon']} & "
                   f"{num_fixed(r['beta'],4)}{stars(pv)} & {num_fixed(r['t'],2)} & {num_fixed(pv,3)} \\\\")
    out += [r"\midrule",
            notes_row(6, extra=("Layer 2 reports cross-factor mean slopes; " + l2_note
                                + "factor-level rows are in the replication outputs.")),
            r"\bottomrule", r"\end{tabular}"]
    write("table3_orth.tex", "\n".join(out) + "\n")


# ---------- Table 4：尾部特征 ----------

def table4() -> None:
    p = REPORTS / "table4_tail.csv"
    if not p.exists():
        log.warning("缺 table4_tail.csv，跳过 Table 4")
        return
    d = pd.read_csv(p)
    cols = [("均值", "Mean"), ("标准差", "SD"), ("偏度", "Skew."),
            ("峰度", "Kurt."), ("崩溃频率", "Crash freq.")]
    out = [r"\begin{tabular}{lll" + "c" * len(cols) + "}", r"\toprule",
           r"Measure & Factor & Crowding & " + " & ".join(c[1] for c in cols) + r" \\",
           r"\midrule"]
    for meas in d["measure"].unique():
        sub = d[d["measure"] == meas]
        for i, (_, r) in enumerate(sub.iterrows()):
            for j, side in enumerate(["高拥挤_", "低拥挤_"]):
                lab = en(meas) if (i == 0 and j == 0) else ""
                out.append(f"{lab} & {en(r['factor'])} & "
                           f"{'High' if j == 0 else 'Low'} & "
                           + " & ".join(num(r[side + c[0]], 4 if c[0] in ("均值", "标准差") else 2)
                                        for c in cols) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    write("table4_tail.tex", "\n".join(out) + "\n", size=r"\footnotesize")


# ---------- Table 7：分布分析（分位数回归 + 高低拥挤组对比，两张合并为一张） ----------

def table7dist() -> None:
    pq = REPORTS / "table4_quantile.csv"
    pt = REPORTS / "table4_tail.csv"
    if not pq.exists() or not pt.exists():
        log.warning("缺 table4_quantile.csv 或 table4_tail.csv，跳过 Table 7")
        return
    dq = pd.read_csv(pq)
    qs = sorted(dq["quantile"].unique())
    pa = [r"\begin{tabular}{l" + "c" * len(qs) + "}", r"\toprule",
          r"\multicolumn{" + str(1 + len(qs)) + r"}{l}{\textit{Panel A: pooled quantile regressions (coefficients)}} \\",
          r"Measure & " + " & ".join(f"$q={q:.2f}$" for q in qs) + r" \\", r"\midrule"]
    for meas in dq["measure"].unique():
        sub = dq[dq["measure"] == meas]
        row = {r["quantile"]: r for _, r in sub.iterrows()}
        cells = [f"{num(row[q]['coef'],4)}{stars(row[q]['p_placebo'])}" if q in row else "—"
                 for q in qs]
        pa.append(f"{en(meas)} & " + " & ".join(cells) + r" \\")
    pa.append(r"\midrule")
    pa.append(r"\multicolumn{" + str(1 + len(qs)) + r"}{l}{\textit{Permutation $p$-values}} \\")
    for meas in dq["measure"].unique():
        sub = dq[dq["measure"] == meas]
        row = {r["quantile"]: r for _, r in sub.iterrows()}
        pa.append(f"{en(meas)} & "
                  + " & ".join(num(row[q]["p_placebo"], 3) if q in row else "—" for q in qs)
                  + r" \\")
    pa += [r"\bottomrule", r"\end{tabular}"]
    write("table4_quantile.tex", "\n".join(pa) + "\n")

    dt = pd.read_csv(pt)
    cols = [("均值", "Mean"), ("标准差", "SD"), ("偏度", "Skew."),
            ("峰度", "Kurt."), ("崩溃频率", "Crash freq.")]
    pb = [r"\begin{tabular}{lll" + "c" * len(cols) + "}", r"\toprule",
          r"\multicolumn{" + str(3 + len(cols)) + r"}{l}{\textit{Panel B: high versus low crowding groups' distributions}} \\",
          r"Measure & Factor & Crowding & " + " & ".join(c[1] for c in cols) + r" \\",
          r"\midrule"]
    for meas in dt["measure"].unique():
        sub = dt[dt["measure"] == meas]
        for i, (_, r) in enumerate(sub.iterrows()):
            for j, side in enumerate(["高拥挤_", "低拥挤_"]):
                lab = en(meas) if (i == 0 and j == 0) else ""
                pb.append(f"{lab} & {en(r['factor'])} & "
                          f"{'High' if j == 0 else 'Low'} & "
                          + " & ".join(num(r[side + c[0]], 4 if c[0] in ("均值", "标准差") else 2)
                                       for c in cols) + r" \\")
        pb.append(r"\midrule")
    pb[-1] = notes_row(3 + len(cols),
                       extra="Panel A reports pooled quantile regressions with factor fixed "
                             "effects; Panel B compares the distributions of the two crowding "
                             "groups.")
    pb += [r"\bottomrule", r"\end{tabular}"]

    body = "\n".join(pa) + "\n\n" + "\n".join(pb)
    write("table7_dist.tex", body, size=r"\footnotesize")


# ---------- Table 7：拥挤度对实现波动/偏度的预测 ----------

def table7() -> None:
    p = REPORTS / "table7_volatility.csv"
    if not p.exists():
        log.warning("缺 table7_volatility.csv，跳过 Table 7")
        return
    d = pd.read_csv(p)
    horizons = sorted(d["horizon"].unique())
    targets = [("波动", "Volatility"), ("偏度", "Skewness")]
    out = [r"\begin{tabular}{llcc}", r"\toprule"]
    for k, (panel, title) in enumerate((("beta", "coefficient"), ("p_placebo", "permutation $p$-value"))):
        if k:
            out.append(r"\midrule")
        out.append(r"\multicolumn{4}{l}{\textit{Panel " + ("A" if k == 0 else "B") +
                   ": " + title + r"}} \\")
        out.append(r"Measure & $h$ & Volatility & Skewness \\")
        out.append(r"\midrule")
        for meas in d["measure"].unique():
            for h in horizons:
                cells = []
                for key, _ in targets:
                    r = d[(d["measure"] == meas) & (d["horizon"] == h) & (d["target"] == key)]
                    if r.empty:
                        cells.append("—")
                        continue
                    r = r.iloc[0]
                    cells.append(f"{num_fixed(r['beta'],4)}{stars(r['p_placebo'])}" if panel == "beta"
                                 else num(r["p_placebo"], 3))
                out.append(f"{en(meas)} & {h} & " + " & ".join(cells) + r" \\")
    out += [r"\midrule", notes_row(4), r"\bottomrule", r"\end{tabular}"]
    write("table7_volatility.tex", "\n".join(out) + "\n")


# ---------- Table 5：稳健性 ----------

def table5() -> None:
    p = REPORTS / "table5_robustness.csv"
    if not p.exists():
        log.warning("缺 table5_robustness.csv，跳过 Table 5")
        return
    d = pd.read_csv(p)
    out = [r"\begin{tabular}{lllcccc}", r"\toprule",
           r"Check & Grouping & Measure & $\beta$ & HAC $t$ & Perm. $p$ & $N$ \\",
           r"\midrule"]
    for _, r in d.iterrows():
        pv = r.get("p_placebo")
        out.append(f"{en(r['检验'])} & {en(r['分组'])} & {en(r['测度'])} & "
                   f"{num_fixed(r['beta'],4)}{stars(pv)} & {num_fixed(r['t'],2)} & {num_fixed(pv,3)} & "
                   f"{int(r['n'])} \\\\")
    out += [r"\midrule", notes_row(7), r"\bottomrule", r"\end{tabular}"]
    write("table5_robust.tex", "\n".join(out) + "\n", size=r"\footnotesize")


# ---------- Table 6：腿分解（16 产出 table8_legs.csv） ----------

def table6() -> None:
    p = REPORTS / "table8_legs.csv"
    if not p.exists():
        log.warning("缺 table8_legs.csv，跳过 Table 6")
        return
    d = pd.read_csv(p)
    out = [r"\begin{tabular}{llllcccc}", r"\toprule",
           r"Panel & Measure & Return leg & $h$ & $\beta$ & HAC $t$ & Perm. $p$ & $N$ \\",
           r"\midrule"]
    # Panel A：同 RHS 精确分解（comomentum 只保留 h=6，h=1 与正文"无腿级模式"一句对应）
    pa = d[(d["analysis"] == "leg decomposition (same RHS)")
           & ~((d["measure"] == "comomentum") & (d["horizon"] == 1))]
    for (meas, h), g in pa.groupby(["measure", "horizon"], sort=False):
        for i, (_, r) in enumerate(g.iterrows()):
            out.append(f"{'A' if i == 0 else ''} & {en(meas)} & {en(r['group'])} & {int(h)} & "
                       f"{num_fixed(r['beta'],4)}{stars(r['p_placebo'])} & {num_fixed(r['t'],2)} & "
                       f"{num_fixed(r['p_placebo'],3)} & {int(r['n'])} \\\\")
    out.append(r"\midrule")
    pb = d[d["analysis"] == "own-leg measure"]
    for (meas, h), g in pb.groupby(["measure", "horizon"], sort=False):
        for i, (_, r) in enumerate(g.iterrows()):
            out.append(f"{'B' if i == 0 else ''} & {en(meas)} & {en(r['group'])} & {int(h)} & "
                       f"{num_fixed(r['beta'],4)}{stars(r['p_placebo'])} & {num_fixed(r['t'],2)} & "
                       f"{num_fixed(r['p_placebo'],3)} & {int(r['n'])} \\\\")
    out += [r"\midrule",
            notes_row(8, extra=("Panel A regresses each leg's horizon-sum return on the "
                                "total margin-balance or comomentum measure, with the "
                                "identical controls and sample as specification D, so the "
                                "long-short coefficient equals the long-leg coefficient "
                                "minus the short-leg coefficient. Panel B pairs each leg's "
                                "return with that leg's own financing measure.")),
            r"\bottomrule", r"\end{tabular}"]
    write("table6_legs.tex", "\n".join(out) + "\n", size=r"\footnotesize", sep="2pt")


# ---------- Table 8：流量型代理（18 产出 table11_flow_proxy.csv） ----------

def table8() -> None:
    p = REPORTS / "table11_flow_proxy.csv"
    if not p.exists():
        log.warning("缺 table11_flow_proxy.csv，跳过 Table 8")
        return
    d = pd.read_csv(p)
    d = d[d["analysis"] == "flow proxy"]
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r"Measure & $h$ & $\beta$ & HAC $t$ & Perm. $p$ & $N$ \\", r"\midrule"]
    for meas in d["measure"].unique():
        sub = d[d["measure"] == meas].sort_values("horizon")
        for i, (_, r) in enumerate(sub.iterrows()):
            out.append(f"{en(meas)} & {int(r['horizon'])} & "
                       f"{num_fixed(r['beta'],4)}{stars(r['p_placebo'])} & {num_fixed(r['t'],2)} & "
                       f"{num_fixed(r['p_placebo'],3)} & {int(r['n'])} \\\\")
        out.append(r"\midrule" if meas != d["measure"].unique()[-1] else "")
    out += [r"\midrule",
            notes_row(6, extra=("Both measures are leg-level float-weighted monthly "
                                "financing intensities (long leg minus short leg) with "
                                "specification-D controls and the measure's own trailing "
                                "twelve-month mean as an additional level control; net "
                                "flows are available for Shanghai only.")),
            r"\bottomrule", r"\end{tabular}"]
    write("table8_flow.tex", "\n".join(out) + "\n")


# ---------- 形成期与收益实现稳健性（26/27 产出） ----------

def table_formation() -> None:
    p = REPORTS / "formation_robustness_regressions.csv"
    if not p.exists():
        log.warning("缺 formation_robustness_regressions.csv，跳过形成期稳健性表")
        return
    d = pd.read_csv(p)
    labels = {"daily_leg_buy_hold": "Frozen-weight daily legs",
              "common_month_end_partial": "Common month-end (partial)"}
    d = d[d["variant"].isin(labels)]
    terminal_path = REPORTS / "terminal_return_regressions.csv"
    terminal_labels = {"raw_realized": "Raw realisations incl. ST days",
                       "terminal_zero_stress": r"Terminal 0\% stress",
                       "terminal_total_loss_stress": r"Terminal -100\% stress"}
    if terminal_path.exists():
        terminal = pd.read_csv(terminal_path)
        terminal = terminal[terminal["variant"].isin(terminal_labels)
                            & terminal["horizon"].eq(6)]
        d = pd.concat([d, terminal], ignore_index=True)
        labels.update(terminal_labels)
    out = [r"\begin{tabular}{lllcccc}", r"\toprule",
           r"Variant & Measure & $h$ & $\beta$ & HAC $t$ & Perm. $p$ & $N$ \\",
           r"\midrule"]
    for variant in labels:
        for meas in ["comomentum", "crowd_margin"]:
            sub = d[(d["variant"] == variant) & (d["measure"] == meas)].sort_values("horizon")
            for i, (_, r) in enumerate(sub.iterrows()):
                out.append(f"{labels[variant] if i == 0 else ''} & {en(meas)} & {int(r['horizon'])} & "
                           f"{num_fixed(r['beta'],4)}{stars(r['p_placebo'])} & {num_fixed(r['t'],2)} & "
                           f"{num_fixed(r['p_placebo'],3)} & {int(r['n'])} \\\\")
    out += [r"\midrule",
            notes_row(7, extra=("Specification-D estimates; production estimates are in the main "
                                "return table. Past-return controls and forward outcomes are "
                                "recomputed from each return series. The daily-leg variant compounds "
                                "available frozen-weight contributions. The partial common-month-end "
                                "variant changes membership and returns but retains production crowding "
                                "and coverage controls. Raw realisations retain subsequent ST trading "
                                "days. Terminal scenarios apply only to unobserved in-holding-month "
                                "terminal events and are not estimated payoffs; all horizons are in the "
                                "accompanying regression CSV.")),
            r"\bottomrule", r"\end{tabular}"]
    write("table9_formation.tex", "\n".join(out) + "\n", size=r"\footnotesize", sep="2pt")


# ---------- 图片导出 ----------

def figures() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    src = REPORTS / "fig_crowding_series.png"
    dst = FIGDIR / "fig_crowding_series.pdf"
    if not src.exists():
        log.warning("缺 fig_crowding_series.png，跳过图片导出")
        return
    try:
        from PIL import Image
        Image.open(src).convert("RGB").save(dst, "PDF", resolution=200.0)
    except Exception as e:  # noqa: BLE001
        log.warning(f"PIL 转换失败（{e}），直接复制 PNG（LaTeX 也能读）")
        shutil.copyfile(src, dst.with_suffix(".png"))
        return
    log.info(f"写出 paper/figures/{dst.name}")


def main() -> None:
    log.info(f"生成论文图表 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")
    table1(); table2(); table3(); table4(); table5(); table6(); table7(); table7dist(); table8()
    table_formation()
    figures()
    heartbeat("12 图表生成结束", log)


if __name__ == "__main__":
    main()
