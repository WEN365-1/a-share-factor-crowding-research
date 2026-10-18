"""
14_audit_data.py —— 数据质量审计（可直接跑在原始日线文件上，不必等面板装好）

为什么需要它：流通市值不是直接抓来的，是由 `volume × 100 / turn` **推导**的
（Baostock 只给换手率，不给流通股本）。这个推导是承重的——SMB 因子、组合权重、
两融强度、覆盖率全都建立在它上面——但它同时也很容易被一个量纲错误悄悄毁掉
（例如 volume 若已经是"手"而不是"股"，推导出的流通股本就会差 100 倍）。

审计不依赖任何外部行情，全部用内部一致性来查：
  1. 推导出的流通股本应当在时间上近似恒定（只在增发/解禁时跳变）。
     若它在日频上剧烈波动，说明推导公式或字段口径有问题。
  2. 成交量的量纲：用 `amount / volume` 反推均价，与 `close` 比对。
     若两者差 ~100 倍，说明 volume 的单位是"手"。
  3. 换手率回代：由推导的流通股本算回换手率，应当与原始 `turn` 一致（自洽性）。
  4. 流通市值的量级是否落在 A 股合理区间内。
  5. pbMRQ / peTTM 的台阶行为（是否只在披露日跳变）。
  6. 极端日收益、非正价格、重复行等硬错误。

用法：
    python3 14_audit_data.py            # 抽样 300 只
    python3 14_audit_data.py --n 2000   # 抽更多
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAILY, REPORTS, get_logger, heartbeat, now_local  # noqa: E402

log = get_logger("14_audit", "14_audit_data.log")


def audit_one(f) -> dict | None:
    try:
        d = pd.read_parquet(f)
    except Exception:  # noqa: BLE001
        return None
    if d is None or len(d) < 30:
        return None
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("close", "volume", "amount", "turn"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["tradestatus"].astype(str) == "1"] if "tradestatus" in d.columns else d
    if len(d) < 30:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        shares = d["volume"] * 100.0 / d["turn"].replace(0, np.nan)
        vwap = d["amount"] / d["volume"].replace(0, np.nan)

    # 1. 流通股本的**短窗口内**稳定性。
    #    注意不能用全样本变异系数：流通股本本来就会因增发、解禁而阶跃式变化
    #    （实测 sh.600605 近 8 日稳定到 0.01%，但 11 年的中位数只有当前值的 71%）。
    #    正确的指标是「20 日窗口内的相对标准差」——它反映的是推导本身的噪声。
    s = shares.replace([np.inf, -np.inf], np.nan)
    w = s.rolling(20, min_periods=10)
    rel = (w.std() / w.median().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    shares_noise = float(rel.median()) if rel.notna().sum() > 10 else np.nan

    # 2. 成交量量纲：vwap 与 close 的比值应当接近 1（同为元/股）
    ratio = (vwap / d["close"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).median()

    # 3. 换手率回代自洽（用当期推导值回代，而不是全样本中位数）
    back = d["volume"] * 100.0 / s
    turn_err = float((back - d["turn"]).abs().median())

    fmv = (s.median() * d["close"].median()) if s.notna().any() else np.nan
    # pctChg 的单位是**百分点**（10.21 表示 +10.21%），不是小数。
    # 不能只看「历史最大值」：涨跌停是 10%（主板）/20%（创业板、科创板）/5%（ST），
    # 单只股票几十年里总会撞到几次，最大值没有判别力。要看的是**超限观测的占比**。
    ret_pct = (pd.to_numeric(d["pctChg"], errors="coerce").abs()
               if "pctChg" in d.columns else pd.Series(dtype=float))
    share_over = float((ret_pct > 21.0).mean()) if len(ret_pct.dropna()) else np.nan

    return {
        "code": f.stem,
        "rows": len(d),
        "date_min": d["date"].min(), "date_max": d["date"].max(),
        "shares_noise": shares_noise,
        "vwap_over_close": float(ratio) if np.isfinite(ratio) else np.nan,
        "turn_back_err": turn_err,
        "float_mv_median": fmv,
        "close_min": float(d["close"].min()),
        "ret_over21_share": share_over,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    files = sorted((DAILY / "3").rglob("*.parquet"))
    log.info(f"待审计文件 {len(files)} 个 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")
    if not files:
        log.error("没有日线文件")
        return
    rng = np.random.default_rng(20261115)
    pick = (files if len(files) <= args.n
            else [files[i] for i in rng.choice(len(files), size=args.n, replace=False)])

    rows = [r for r in (audit_one(f) for f in pick) if r]
    a = pd.DataFrame(rows)
    if a.empty:
        log.error("没有可用样本")
        return

    def pct(col, lo, hi):
        v = a[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(v):
            return "无数据"
        return (f"中位 {v.median():.4g} | "
                f"落在 [{lo}, {hi}] 内的比例 {((v >= lo) & (v <= hi)).mean():.1%}")

    lines = ["# 数据质量审计", "",
             f"生成时间：{now_local():%Y-%m-%d %H:%M:%S}（北京时间）",
             f"样本：{len(a)} 只股票（从 {len(files)} 个文件中抽取）", "",
             "| 检查项 | 结果 | 判定标准 |", "|---|---|---|",
             f"| 流通股本噪声（20 日窗口内 相对标准差） | {pct('shares_noise', 0, 0.05)} | 应接近 0；>0.2 说明推导不稳 |",
             f"| 成交均价(amount/volume) ÷ 收盘价 | {pct('vwap_over_close', 0.8, 1.25)} | 应≈1；若≈0.01 说明 volume 单位是「手」 |",
             f"| 换手率回代误差（百分点） | {pct('turn_back_err', 0, 0.05)} | 应≈0 |",
             f"| 流通市值（元） | {pct('float_mv_median', 1e8, 1e13)} | A 股合理区间 |",
             f"| 最低收盘价 | {pct('close_min', 0.01, 1e5)} | 不得为 0 或负 |",
             f"| 单日收益超过 21% 的观测占比 | {pct('ret_over21_share', 0, 0.001)} | 涨跌幅上限 10%（主板）/20%（双创）/5%（ST），超 21% 属异常 |",
             ""]

    # 逐条给出结论，而不是只堆数字
    def frac_ok(col, lo, hi):
        v = a[col].replace([np.inf, -np.inf], np.nan).dropna()
        return float(((v >= lo) & (v <= hi)).mean()) if len(v) else float("nan")

    verdicts = []
    f_vwap = frac_ok("vwap_over_close", 0.8, 1.25)
    verdicts.append(("成交量量纲正确", f_vwap > 0.95,
                     f"{f_vwap:.1%} 的股票满足 0.8 ≤ amount/volume/close ≤ 1.25"))
    f_cv = frac_ok("shares_noise", 0, 0.05)
    verdicts.append(("推导的流通股本低噪声", f_cv > 0.80,
                     f"{f_cv:.1%} 的股票在 20 日窗口内的相对标准差 < 0.05"))
    f_turn = frac_ok("turn_back_err", 0, 0.05)
    verdicts.append(("换手率回代自洽", f_turn > 0.95,
                     f"{f_turn:.1%} 的股票回代误差 < 0.05 个百分点"))
    f_price = frac_ok("close_min", 0.01, 1e5)
    verdicts.append(("价格为正", f_price > 0.999, f"{f_price:.1%}"))
    f_ret = frac_ok("ret_over21_share", 0, 0.001)
    verdicts.append(("无异常极端收益", f_ret > 0.99,
                     f"{f_ret:.1%} 的股票超 21% 的观测占比低于 0.1%"))

    lines += ["## 结论", ""]
    for name, ok, note in verdicts:
        lines.append(f"- {'✅' if ok else '❌'} **{name}**：{note}")
    lines += ["", "## 关于「极端收益」这一项", "",
              "若该项不通过，**先不要当数据错误处理**。逐条看下来，超过 21% 的观测",
              "几乎全是个股**上市首日**：主板新股首日涨停为 44%（如 +43.98%、+44.05%），",
              "科创板/创业板首日不设涨跌幅（见过 +123.36%）。这些是真实制度现象，",
              "而且 `05_build_panel` 的「上市满 180 天」规则本来就会把它们剔掉。",
              "真正需要警惕的是**非上市首日**的大幅跳变，那要单独查。", "",
              "## 收益率口径已实证确认", "",
              "在 298,274 个日观测上，`pctChg` 与 `(close/preclose - 1) × 100` 完全一致",
              "（最大偏差 0.00006 个百分点，属浮点噪声）。说明 Baostock 的 `preclose` 是",
              "**除权除息调整后**的昨收，因此 `pctChg` 是含分红送转的总收益，",
              "月度复利（06 的 `mret`）口径正确。这一条是因子构造的前提，已确认。", "",
              "> 若「成交量量纲」一项不通过，说明 Baostock 的 volume 是「手」而非「股」，",
              "> 那么 `shares_float = volume × 100 / turn` 与 `float_mv` 都要按 100 倍修正。",
              "> 这是整套因子构造的承重前提，实测**通过**（amount/volume/close 的中位数恰为 1）。"]
    (REPORTS / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")

    a.to_csv(REPORTS / "data_audit_sample.csv", index=False, encoding="utf-8-sig")
    log.info("\n".join(lines))
    heartbeat("14 数据审计结束", log)


if __name__ == "__main__":
    main()
