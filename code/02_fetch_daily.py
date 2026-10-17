"""
02_fetch_daily.py —— 全 A 日线面板（Baostock，断点续传）

产出：data/raw/daily/3/<code>.parquet     不复权日线 + 估值字段
      data/raw/daily/2/<code>.parquet     前复权日线（可选，用于收益率口径校验）

为什么主拉「不复权」：
  不复权口径下 价 / 量 / 换手率 / 估值字段 全部自洽。收益率用 Baostock 的 pctChg
  字段（该字段以交易所除权除息后的昨收为基准，已包含分红送转，是真实收益率），
  因此不必为了收益率再拉一份前复权；日常估值因子用 pbMRQ/peTTM 这类比值字段，
  与价格是否复权无关。一份数据同时满足收益、市值、估值三类用途。

流通市值推导：换手率(%) = 成交量(股) / 流通股本 × 100
            ⇒ 流通市值 = 成交量 / (换手率/100) × 收盘价
  停牌日换手率为 0，用前值填充（在 06_build_panel 里做）。
"""
from __future__ import annotations

import argparse
import sys
import time
import pandas as pd
import baostock as bs

sys.path.insert(0, "/home/user/金融科研1014/code")
from common import (CLEAN, CFG, DAILY, Gate, Progress, bs_login_ok, get_logger,  # noqa: E402
                    heartbeat, read_parquet, unit_path, write_parquet)

log = get_logger("02_daily", "02_daily.log")


def fetch_one(code: str, start: str, end: str, fields: str) -> pd.DataFrame | None:
    rs = bs.query_history_k_data_plus(code, fields, start_date=start, end_date=end,
                                      frequency="d", adjustflag="3")
    if rs.error_code != "0":
        raise RuntimeError(f"error_code={rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    return pd.DataFrame(rows, columns=rs.fields)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0, help="本进程负责的分片编号（从 0 开始）")
    ap.add_argument("--nshards", type=int, default=1, help="总分片数")
    args = ap.parse_args()

    cfg = CFG
    gate = Gate(log)
    start, end = cfg["universe"]["start"], cfg["universe"]["end"]
    fields = cfg["fetch"]["daily_fields"]
    retry = cfg["fetch"]["retry"]
    sleep_s = cfg["fetch"]["request_sleep"]

    uni = read_parquet(CLEAN / "universe.parquet")
    all_codes = uni["code"].tolist()
    codes = [c for i, c in enumerate(all_codes) if i % args.nshards == args.shard]
    tag = f"daily#{args.shard}/{args.nshards}"
    log.info(f"[分片 {args.shard}/{args.nshards}] 待拉取 {len(codes)}/{len(all_codes)} 只 | 字段 {fields}")
    heartbeat(f"02 日线面板开始 {tag}", log)

    if not bs_login_ok(bs, log):
        sys.exit(1)

    prog = Progress(len(codes), log, tag, every=cfg["fetch"]["progress_every"])
    base = DAILY / "3"
    t_first = None
    failures: list[str] = []

    for i, code in enumerate(codes):
        if not gate.ok():
            log.warning(f"闸门触发，日线阶段 {tag} 在第 {i} 只处收手（进度已落盘，重启即续）")
            break
        p = unit_path(base, code)
        if p.exists() and p.stat().st_size > 0:
            prog.tick(skipped=True)
            continue

        df = None
        for attempt in range(retry):
            try:
                df = fetch_one(code, start, end, fields)
                break
            except Exception as e:  # noqa: BLE001
                wait = cfg["fetch"]["retry_sleep"] * (2 ** attempt)
                log.warning(f"{code} 第 {attempt + 1}/{retry} 次失败：{e}；{wait:.0f}s 后重试")
                time.sleep(wait)
                try:
                    bs.logout()
                except Exception:  # noqa: BLE001
                    pass
                if not bs_login_ok(bs, log, tries=6):
                    break
        else:
            failures.append(code)

        if df is not None and len(df):
            df["adjustflag"] = "3"
            write_parquet(df, p)
            if t_first is None:
                t_first = time.time()
        elif df is None and code not in failures:
            # 合法空结果（如退市早于样本期）：写空占位符，避免下次重拉
            write_parquet(pd.DataFrame(columns=fields.split(",") + ["adjustflag"]), p)

        time.sleep(sleep_s)
        prog.tick()

        if i == 20 and t_first:
            el = time.time() - t_first
            rate = el / 21
            log.info(f"★ {tag} 速度：{rate:.2f} s/只 → 本分片 {len(codes)} 只预计 {rate * len(codes) / 3600:.1f} 小时")

    prog.done()
    if failures:
        log.warning(f"[{tag}] 最终失败 {len(failures)} 只：{failures[:30]}")
    heartbeat(f"02 日线面板结束 {tag}", log)
    bs.logout()
    log.info("阶段 02 完成")


if __name__ == "__main__":
    main()
