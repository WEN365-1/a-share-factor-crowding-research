"""
check_fetch.py —— 抓取完整性校验

为什么需要它：分片进程会因为偶发网络错误中途退出。实测发生过一次——日线分片 5 的
Baostock 登录返回"网络接收错误"，脚本直接 `sys.exit(1)`，**静默丢掉 678 只股票**。
run_all.sh 原来只等进程结束就继续往下跑，于是 05 会在一份不完整的面板上装配，
而且日志里没有任何显眼的报错。

用法：
    python3 check_fetch.py
完整 → 退出码 0；不完整 → 退出码 1，并把缺失清单打到 stdout（run_all.sh 据此重试）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/user/金融科研1014/code")
from common import CFG, CLEAN, DAILY, MARGIN, read_parquet  # noqa: E402


def main() -> int:
    problems = []

    # ---------- 日线：一股票一文件（合法空结果也会写占位文件）----------
    # 注意用 rglob：common.unit_path 会按 key 前 6 位再分一层子目录（sh.603139 → sh.603/）
    uni = read_parquet(CLEAN / "universe.parquet")
    want = set(uni["code"].astype(str))
    have = {p.stem for p in (DAILY / "3").rglob("*.parquet")}
    miss_d = want - have
    print(f"日线：应有 {len(want)} 只，已落 {len(have)} 只，缺 {len(miss_d)} 只")
    if miss_d:
        problems.append(f"日线缺 {len(miss_d)} 只，例如 {sorted(miss_d)[:10]}")

    # ---------- 两融：一「交易日 × 交易所」一文件 ----------
    cal = read_parquet(CLEAN / "calendar.parquet")
    days = list(pd.to_datetime(cal["date"]))
    exs = list(CFG["margin"]["exchanges"])
    want_m = {(ex, f"{d:%Y%m%d}") for ex in exs for d in days}
    have_m = set()
    empty_m = []
    bad_m = []
    required = {"date", "code", "margin_balance"}
    for ex in exs:
        for p in (MARGIN / ex).rglob("*.parquet"):
            key = (ex, p.stem)
            have_m.add(key)
            try:
                cols = set(pd.read_parquet(p, columns=None).columns)
                if not required.issubset(cols):
                    bad_m.append((ex, p.stem, sorted(required - cols)))
                    continue
                if p.stat().st_size <= 0 or len(pd.read_parquet(p, columns=["code"])) == 0:
                    empty_m.append(key)
            except Exception as e:  # noqa: BLE001
                bad_m.append((ex, p.stem, [type(e).__name__]))
    miss_m = want_m - have_m
    print(f"两融：应有 {len(want_m)} 个交易日 × 交易所单位，已落 {len(have_m)} 个，缺 {len(miss_m)} 个；"
          f"空文件 {len(empty_m)} 个，字段异常 {len(bad_m)} 个")
    if miss_m:
        problems.append(f"两融缺 {len(miss_m)} 个单位，例如 {sorted(miss_m)[:5]}")
    if empty_m:
        problems.append(f"两融空文件 {len(empty_m)} 个，例如 {sorted(empty_m)[:5]}")
    if bad_m:
        problems.append(f"两融字段异常 {len(bad_m)} 个，例如 {bad_m[:3]}")

    if problems:
        print("\n不完整：")
        for p in problems:
            print("  -", p)
        return 1
    print("\n抓取完整。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
