"""
tests/make_synthetic.py —— 生成一套结构逼真的合成数据，用于在真实数据到达前
                           把全部分析脚本（06/07/08/09/10/11）跑通，提前暴露 bug。

合成数据覆盖真实的 schema 与边界情况：
  - 面板按年分区、字段名与真实面板一致
  - 两融明细**只覆盖一半股票**，复现「只有标的股才有数据」这一真实特征，
    以及由此必须处理的「覆盖率」问题
  - 部分行业标为「银行/证券」，用于验证剔除规则

**本合成数据不含真实的拥挤→衰减效应**，所以它验证的是代码路径、数据形状与数值合理性，
不验证统计功效。回归模块能否测出已知效应，由 tests/test_regression_recovery.py 单独验证。

用法：
    FIN_ROOT=/tmp/finsmoke python3 tests/make_synthetic.py
    FIN_ROOT=/tmp/finsmoke python3 06_build_factors.py
    ... 依次跑 07 08 09 10 11
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(os.environ.get("FIN_ROOT", "/tmp/finsmoke"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN, DATA, MARGIN, RAW, ensure_dirs, write_parquet  # noqa: E402

# 种子可用环境变量覆盖：判断某个结果是真模式还是这一抽的偶然，换种子重跑是最直接的办法。
RNG = np.random.default_rng(int(os.environ.get("SYNTH_SEED", "42")))
# 必须大到每个十分位有足够股票：08 的两两相关要求每端至少 MIN_PER_LEG=15 只，
# 200 只股票时每个十分位只有 13 只，comomentum 会被整表跳过，冒烟测试就测了个寂寞。
N_STOCKS = 800
START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2024-12-31")
TRADE_DAYS = pd.bdate_range(START, END)
INDUSTRIES = ["制造业", "信息技术", "医药生物", "银行", "证券", "食品饮料"]


def main() -> None:
    ensure_dirs()
    codes = [f"sh.6{i:05d}" for i in range(N_STOCKS // 2)] + \
            [f"sz.0{i:05d}" for i in range(N_STOCKS // 2)]
    print(f"合成 {len(codes)} 只股票 × {len(TRADE_DAYS)} 个交易日 → {ROOT}")

    # ---------- 日频面板 ----------
    # 每只股票给一个固定的截面特征值（决定因子暴露），再叠加日频噪声
    char = pd.DataFrame({
        "code": codes,
        "bm": RNG.normal(0, 1, len(codes)),          # 价值
        "roe": RNG.normal(0, 1, len(codes)),         # 质量
        "vol": RNG.uniform(0.01, 0.05, len(codes)),  # 波动
        "turn": RNG.uniform(0.5, 8.0, len(codes)),   # 换手
        "mv": np.exp(RNG.normal(22, 1.2, len(codes))),
        "industry": RNG.choice(INDUSTRIES, len(codes)),
    })
    char["beta_mkt"] = RNG.normal(1.0, 0.3, len(codes))
    char["beta_smb"] = RNG.normal(0.3, 0.5, len(codes))
    char["beta_hml"] = RNG.normal(0.2, 0.4, len(codes))
    char["id"] = RNG.normal(0, 0.004, len(codes))     # 个股特质收益

    mkt = RNG.normal(0.0002, 0.010, len(TRADE_DAYS))
    smb = RNG.normal(0.0000, 0.005, len(TRADE_DAYS))
    hml = RNG.normal(0.0000, 0.005, len(TRADE_DAYS))

    frames = []
    for i, row in char.iterrows():
        eps = RNG.normal(row["id"] * 0.5, row["vol"], len(TRADE_DAYS))
        ret = (row["beta_mkt"] * mkt + row["beta_smb"] * smb + row["beta_hml"] * hml + eps)
        px = 10 * np.exp(np.cumsum(ret))
        frames.append(pd.DataFrame({
            "date": TRADE_DAYS, "code": row["code"],
            "close": px.astype("float32"),
            "ret": ret.astype("float32"),
            "volume": (row["mv"] / px * row["turn"] / 100).astype("float32"),
            "amount": (row["mv"] / px * row["turn"] / 100 * px).astype("float32"),
            "turn": np.full(len(TRADE_DAYS), row["turn"], dtype="float32"),
            "float_mv": np.full(len(TRADE_DAYS), row["mv"], dtype="float32"),
            "shares_float": (row["mv"] / px).astype("float32"),
            "industry": row["industry"],
            "peTTM": np.full(len(TRADE_DAYS), np.exp(row["roe"]) * 15, dtype="float32"),
            "pbMRQ": np.full(len(TRADE_DAYS), np.exp(row["bm"]) * 2, dtype="float32"),
            "psTTM": np.full(len(TRADE_DAYS), 2.0, dtype="float32"),
            "pcfNcfTTM": np.full(len(TRADE_DAYS), 10.0, dtype="float32"),
        }))
    # ---------- 原始日线（Baostock 口径），供 05_build_panel 走完整流程 ----------
    #
    # **必须制造停牌**：让一部分股票在随机月份的月末最后 1~3 个交易日没有数据。
    # 这是真实数据的关键特征——停牌股当月的「最后交易日」因此各不相同，
    # 而 08 的 month_ends 若从个股快照里收集，就会同一个月冒出几十个端点、
    # 把每个 (月, 因子) 算重、最后把合并表炸开。
    # 合成数据原先因为每只股票每个交易日都有数据、每月只有一个快照日，
    # **完全漏掉了这个 bug**（真金白银的一次教训）。
    panel = pd.concat(frames, ignore_index=True)
    panel["_ym"] = panel["date"].dt.to_period("M")
    susp_codes = set(RNG.choice(codes, size=max(1, int(0.20 * len(codes))), replace=False))
    drop_idx = []
    for code in susp_codes:
        g = panel[panel["code"] == code]
        months = g["_ym"].unique()
        picked = RNG.choice(months, size=max(1, int(0.25 * len(months))), replace=False)
        for ym in picked:
            sub = g[g["_ym"] == ym]
            k = int(RNG.integers(1, 4))
            drop_idx.extend(sub.index[-k:].tolist())
    if drop_idx:
        panel = panel.drop(index=drop_idx).reset_index(drop=True)
        print(f"  已制造停牌：{len(susp_codes)} 只股票、共 {len(drop_idx)} 个交易日缺失")
    panel = panel.drop(columns=["_ym"])

    for code, g in panel.groupby("code"):
        raw = pd.DataFrame({
            "date": g["date"].dt.strftime("%Y-%m-%d").values,
            "code": code,
            "open": g["close"].values, "high": g["close"].values,
            "low": g["close"].values, "close": g["close"].values,
            "preclose": (g["close"] / (1 + g["ret"])).values,
            "volume": g["volume"].values, "amount": g["amount"].values,
            "turn": g["turn"].values, "tradestatus": "1", "pctChg": (g["ret"] * 100).values,
            "isST": "0", "peTTM": g["peTTM"].values, "pbMRQ": g["pbMRQ"].values,
            "psTTM": g["psTTM"].values, "pcfNcfTTM": g["pcfNcfTTM"].values,
        })
        write_parquet(raw, RAW / "daily" / "3" / f"{code}.parquet")
    print(f"  原始日线 {len(panel):,} 行 → {RAW / 'daily' / '3'}")

    # ---------- 两融明细（只覆盖一半股票，模拟「只有标的股才有数据」）----------
    mg_codes = char["code"].sample(N_STOCKS // 2, random_state=1).tolist()
    mb = {c: RNG.uniform(1e7, 5e8) for c in mg_codes}
    mg_rows = []
    for d in TRADE_DAYS:
        for c in mg_codes:
            mb[c] *= (1 + RNG.normal(0.0008, 0.01))   # 余额随机游走
            mg_rows.append({"date": d, "exchange": "sse" if c.startswith("sh") else "szse",
                            "code": c, "name": c, "margin_balance": mb[c],
                            "fin_buy": mb[c] * 0.05, "short_vol": 0.0})
    mg = pd.DataFrame(mg_rows)
    for d, g in mg.groupby("date"):
        ex = "sse" if g["exchange"].iloc[0] == "sse" else "szse"
        write_parquet(g, MARGIN / ex / f"{d:%Y%m%d}.parquet")
    print(f"  两融明细 {len(mg):,} 行 → {MARGIN}")

    # ---------- 股票池 / 日历 / 行业 ----------
    write_parquet(pd.DataFrame({
        "code": char["code"], "code_name": char["code"],
        "ipoDate": pd.Timestamp("2015-01-01"), "outDate": pd.NaT,
        "type": "1", "status": "1", "exchange": char["code"].str[:2],
    }), CLEAN / "universe.parquet")
    write_parquet(pd.DataFrame({"date": TRADE_DAYS}), CLEAN / "calendar.parquet")
    write_parquet(pd.DataFrame({"code": char["code"], "industry": char["industry"]}),
                  CLEAN / "industry.parquet")
    print("  股票池 / 日历 / 行业 已写入")


if __name__ == "__main__":
    main()
