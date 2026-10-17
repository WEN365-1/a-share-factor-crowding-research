"""
03_fetch_margin.py —— 个股两融明细全量（AKShare，断点续传）

产出：data/raw/margin/sse/<YYYYMMDD>.parquet
      data/raw/margin/szse/<YYYYMMDD>.parquet

实操三个坑（已写进代码）：
  1. 明细接口按单日取数 → 必须按交易日循环，2674 日 × 2 所 ≈ 5348 次请求
  2. 非交易日 / 瞬时故障会返回空表 → 用交易日历过滤 + 空表也写占位符，避免每次重拉
  3. 沪深字段集不一致 → 这里只做归一化落盘，口径统一留到 05_build_panel

字段映射（两所统一到同一套列名）：
  SSE:  信用交易日期 标的证券代码 标的证券简称 融资余额 融资买入额 融资偿还额 融券余量 融券卖出量 融券偿还量
  SZSE: 证券代码 证券简称 融资买入额 融资余额 融券卖出量 融券余量 融券余额 融资融券余额
"""
from __future__ import annotations

import json
import sys
import time
import pandas as pd
import akshare as ak
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/user/金融科研1014/code")
from common import (CLEAN, CFG, MARGIN, Gate, Progress, get_logger, heartbeat,  # noqa: E402
                    read_parquet, unit_path, write_parquet)

log = get_logger("03_margin", "03_margin.log")

SSE_MAP = {"标的证券代码": "code", "标的证券简称": "name", "融资余额": "margin_balance",
           "融资买入额": "fin_buy", "融资偿还额": "fin_repay", "融券余量": "short_vol",
           "融券卖出量": "short_sell", "融券偿还量": "short_repay"}
SZSE_MAP = {"证券代码": "code", "证券简称": "name", "融资余额": "margin_balance",
            "融资买入额": "fin_buy", "融券余量": "short_vol", "融券卖出量": "short_sell",
            "融券余额": "short_balance", "融资融券余额": "total_balance"}


def norm(df: pd.DataFrame, mapping: dict, ex: str, date: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "exchange", "code", "name", "margin_balance",
                                     "fin_buy", "short_vol"])
    out = df.rename(columns=mapping)
    keep = [c for c in ["code", "name", "margin_balance", "fin_buy", "fin_repay",
                        "short_vol", "short_sell", "short_repay", "short_balance",
                        "total_balance"] if c in out.columns]
    out = out[keep].copy()
    for c in keep:
        if c not in ("code", "name"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["date"] = pd.to_datetime(date, format="%Y%m%d")
    out["exchange"] = ex
    # 统一代码格式为 sh.600000 / sz.000001
    out["code"] = out["code"].astype(str).str.zfill(6)
    pre = "sh" if ex == "sse" else "sz"
    out["code"] = pre + "." + out["code"]
    return out


def main() -> None:
    cfg = CFG
    gate = Gate(log)
    retry = cfg["margin"]["retry"]
    sleep_s = cfg["margin"]["request_sleep"]

    cal = read_parquet(CLEAN / "calendar.parquet")
    days = pd.to_datetime(cal["date"]).dt.strftime("%Y%m%d").tolist()
    jobs = [(d, ex) for d in days for ex in cfg["margin"]["exchanges"]]
    log.info(f"待拉取 {len(jobs)} 个单位（{len(days)} 交易日 × {len(cfg['margin']['exchanges'])} 所）")
    heartbeat("03 两融明细开始", log)

    prog = Progress(len(jobs), log, "margin", every=100)
    t0 = time.time()
    done_n = 0
    failures = []

    for i, (d, ex) in enumerate(jobs):
        if not gate.ok():
            log.warning(f"闸门触发，两融阶段在第 {i} 个单位处收手（进度已落盘，重启即续）")
            break
        base = MARGIN / ex
        p = unit_path(base, d)
        if p.exists() and p.stat().st_size > 0:
            try:
                existing = pd.read_parquet(p, columns=["date", "code", "margin_balance"])
                if len(existing) and existing["code"].notna().any():
                    prog.tick(skipped=True)
                    continue
            except Exception:  # noqa: BLE001
                pass

        fn = ak.stock_margin_detail_sse if ex == "sse" else ak.stock_margin_detail_szse
        df = None
        for attempt in range(retry):
            try:
                df = fn(date=d)
                break
            except Exception as e:  # noqa: BLE001
                wait = cfg["margin"]["retry_sleep"] * (2 ** attempt)
                log.warning(f"{ex} {d} 第 {attempt + 1}/{retry} 次失败：{str(e)[:80]}；{wait:.0f}s 后重试")
                time.sleep(wait)
        else:
            failures.append((ex, d))

        if df is None:
            # 失败不能写成看似成功的空 parquet；否则完整性检查会把失败误判为完成。
            fail_path = p.with_suffix(p.suffix + ".failed.json")
            fail_path.write_text(json.dumps({
                "exchange": ex, "date": d, "status": "failed"
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            log.error(f"{ex} {d} 请求失败，已写失败标记 {fail_path.name}")
            time.sleep(sleep_s)
            prog.tick()
            continue

        out = norm(df, SSE_MAP if ex == "sse" else SZSE_MAP, ex, d)
        if out.empty:
            log.warning(f"{ex} {d} 返回空表；不写入成功文件，下一轮继续重试")
            failures.append((ex, d, "empty"))
            time.sleep(sleep_s)
            prog.tick()
            continue
        write_parquet(out, p)
        done_n += 1

        time.sleep(sleep_s)
        prog.tick()

        if i == 30:
            el = time.time() - t0
            rate = el / 31
            log.info(f"★ 速度标定：约 {rate:.2f} s/次 → 全量预计 {rate * len(jobs) / 3600:.1f} 小时")

    prog.done()
    if failures:
        log.warning(f"最终失败 {len(failures)} 个：{failures[:20]}")
    heartbeat("03 两融明细结束", log)
    log.info("阶段 03 完成")


if __name__ == "__main__":
    main()
