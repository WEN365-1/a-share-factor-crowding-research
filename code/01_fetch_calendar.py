"""
01_fetch_calendar.py —— 股票池与交易日历

产出：
  data/clean/universe.parquet   全 A 股票基本资料（含退市股，规避幸存者偏差）
  data/clean/calendar.parquet   2015-2025 交易日历
  data/clean/industry.parquet   行业分类

关键点：用 query_stock_basic() 取「所有出现过」的股票（含已退市），
      而不是 query_all_stock(某个日期) —— 后者只给当天在市的，会引入幸存者偏差。
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import baostock as bs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, CFG, Gate, META, bs_login_ok, get_logger, load_config,  # noqa: E402
                    write_parquet, now_local)

log = get_logger("01_calendar", "01_calendar.log")


def main() -> None:
    gate = Gate(log)
    cfg = CFG
    start, end = cfg["universe"]["start"], cfg["universe"]["end"]
    if not bs_login_ok(bs, log):
        sys.exit(1)
    log.info(f"Baostock 登录成功 | 样本期 {start} ~ {end} | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    # ---------- 1. 股票池（含退市股） ----------
    rs = bs.query_stock_basic()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    uni = pd.DataFrame(rows, columns=rs.fields)
    log.info(f"query_stock_basic 返回 {len(uni)} 条，字段 {list(uni.columns)}")

    # type: 1=股票 2=指数 3=其它 4=可转债 5=ETF；status: 1=上市 0=退市
    uni = uni[uni["type"] == "1"].copy()
    uni["exchange"] = uni["code"].str.split(".").str[0]
    uni = uni[uni["exchange"].isin(cfg["universe"]["exchanges"])].copy()
    uni["ipoDate"] = pd.to_datetime(uni["ipoDate"], errors="coerce")
    uni["outDate"] = pd.to_datetime(uni["outDate"], errors="coerce")
    # 与样本期有交集的才算进池
    uni = uni[(uni["outDate"].isna()) | (uni["outDate"] >= pd.Timestamp(start))]
    uni = uni[uni["ipoDate"] <= pd.Timestamp(end)].copy()
    write_parquet(uni, CLEAN / "universe.parquet")
    log.info(
        f"股票池：{len(uni)} 只（含已退市 {(uni['status'] == '0').sum()} 只）"
        f" | 沪 {(uni['exchange'] == 'sh').sum()} 深 {(uni['exchange'] == 'sz').sum()}"
    )

    # ---------- 2. 交易日历 ----------
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    cal = pd.DataFrame(rows, columns=rs.fields)
    cal["calendar_date"] = pd.to_datetime(cal["calendar_date"])
    cal["is_trading_day"] = cal["is_trading_day"].astype(str) == "1"
    trade_days = cal[cal["is_trading_day"]].copy()
    write_parquet(trade_days[["calendar_date"]].rename(columns={"calendar_date": "date"}),
                  CLEAN / "calendar.parquet")
    log.info(f"交易日历：{len(cal)} 个自然日，其中交易日 {len(trade_days)} 个")

    # ---------- 3. 行业分类 ----------
    rs = bs.query_stock_industry()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    ind = pd.DataFrame(rows, columns=rs.fields)
    write_parquet(ind, CLEAN / "industry.parquet")
    log.info(f"行业分类：{len(ind)} 条；行业数 {ind['industry'].nunique() if len(ind) else 0}")

    # ---------- 4. 落盘一份元信息，供后续阶段读取 ----------
    import json
    meta = {
        "n_stocks": int(len(uni)),
        "n_trading_days": int(len(trade_days)),
        "start": start,
        "end": end,
        "built_at_local": now_local().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (META / "universe_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    bs.logout()
    log.info("阶段 01 完成")


if __name__ == "__main__":
    main()
