"""Multi-stock point-in-time audit for Baostock valuation fields.

The audit asks whether implied BPS/EPS steps occur around financial-report
publication dates rather than report-period ends. It is a diagnostic on a
stratified sample, not proof that the provider can never revise history.

Outputs:
  reports/pit_audit_multistock.csv
  reports/pit_audit_multistock_steps.csv
  reports/pit_audit_failures.csv
  reports/pit_audit_multistock.md

Raw query responses are cached per stock under data/clean/pit_audit_cache so an
interrupted network run can resume and the exact audited responses remain
inspectable. Use --refresh to replace the cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN, REPORTS, get_logger, heartbeat, now_local  # noqa: E402

log = get_logger("17_pit", "17_pit_audit.log")

N_STOCKS = 40
SEED = 20261018
START, END = "2015-01-01", "2025-12-31"
BPS_STEP = 0.003
EPS_STEP = 0.005
TOL_LO, TOL_HI = 2, 4  # Trading-day indices, not calendar days.
QE_TOL = 1
CACHE = CLEAN / "pit_audit_cache"
PROFIT_COLUMNS = ["code", "pubDate", "statDate", "roeAvg", "npMargin", "gpMargin",
                  "netProfit", "epsTTM", "MBRevenue", "totalShare", "liqaShare"]


def _result_rows(rs) -> list[list[str]]:
    if rs.error_code != "0":
        raise RuntimeError(f"Baostock error {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return rows


def query_k(code: str) -> list[list[str]]:
    import baostock as bs
    for attempt in range(4):
        try:
            rs = bs.query_history_k_data_plus(code, "date,close,pbMRQ,peTTM",
                                              start_date=START, end_date=END,
                                              frequency="d", adjustflag="3")
            rows = _result_rows(rs)
            if rows:
                return rows
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            log.warning(f"{code} K 线查询重试：{exc}")
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{code} K 线查询为空")


def query_profit(code: str, start_year: int = 2015) -> list[list[str]]:
    """Query one report period at a time, including the previous annual report."""
    import baostock as bs
    rows = []
    for year in range(max(start_year - 1, 2014), 2026):
        for quarter in (1, 2, 3, 4):
            for attempt in range(3):
                try:
                    rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
                    rows.extend(_result_rows(rs))
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        raise
                    log.warning(f"{code} {year}Q{quarter} 财报日期查询重试：{exc}")
                    time.sleep(1.0 * (attempt + 1))
    return rows


def cached_queries(code: str, start_year: int, refresh: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{code.replace('.', '_')}.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if int(cached.get("start_year", 9999)) <= start_year:
            return cached
    payload = {"code": code, "start_year": start_year,
               "k": query_k(code), "profit": query_profit(code, start_year)}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return payload


def steps_of(series: pd.Series, threshold: float) -> list[int]:
    """Return starts of runs whose adjacent relative change exceeds threshold."""
    values = series.to_numpy(dtype=float)
    if len(values) < 2:
        return []
    hits = np.zeros(len(values), dtype=bool)
    previous = values[0]
    for i in range(1, len(values)):
        if previous > 0 and values[i] > 0 and abs(values[i] - previous) / previous > threshold:
            hits[i] = True
        previous = values[i]
    return [i for i, hit in enumerate(hits) if hit and (i == 0 or not hits[i - 1])]


def _nearest_publication(step_index: int, dates: pd.Series, publications: np.ndarray):
    """Match a step to publications in its [-2,+4] trading-day neighbourhood."""
    if not publications.size:
        return False, pd.NaT, np.nan
    lo = dates.iloc[max(0, step_index - TOL_LO)]
    hi = dates.iloc[min(len(dates) - 1, step_index + TOL_HI)]
    eligible = publications[(publications >= np.datetime64(lo)) & (publications <= np.datetime64(hi))]
    if not eligible.size:
        return False, pd.NaT, np.nan
    # Distance is measured in trading-row indices to the first trading day on/after publication.
    distances = [abs(step_index - int(dates.searchsorted(pd.Timestamp(pub)))) for pub in eligible]
    chosen = eligible[int(np.argmin(distances))]
    return True, pd.Timestamp(chosen), float(min(distances))


def _near_quarter_end(step_index: int, dates: pd.Series, quarter_ends: np.ndarray) -> bool:
    if not quarter_ends.size:
        return False
    for quarter_end in quarter_ends:
        centre = int(dates.searchsorted(pd.Timestamp(quarter_end)))
        if any(abs(step_index - candidate) <= QE_TOL
               for candidate in (centre - 1, centre, centre + 1)):
            return True
    return False


def audit_frames(code: str, k: pd.DataFrame, profit: pd.DataFrame) -> tuple[dict | None, pd.DataFrame]:
    """Pure audit implementation, separated from network access for unit testing."""
    if len(k) < 400:
        return None, pd.DataFrame()
    k = k.copy()
    for column in ("close", "pbMRQ", "peTTM"):
        k[column] = pd.to_numeric(k[column], errors="coerce")
    k["date"] = pd.to_datetime(k["date"], errors="coerce")
    k = k.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if len(k) < 400 or k["date"].duplicated().any():
        return None, pd.DataFrame()
    dates = k["date"]
    k["bps"] = np.where(k["pbMRQ"] > 0.01, k["close"] / k["pbMRQ"], np.nan)
    # Negative or near-zero PE has no economically useful EPS reciprocal; this diagnostic
    # therefore covers positive-PE observations only and is not a full-history EPS audit.
    k["eps"] = np.where(k["peTTM"] > 0.5, k["close"] / k["peTTM"], np.nan)

    publications = np.asarray(sorted(pd.to_datetime(
        profit.get("pubDate", pd.Series(dtype=object)), errors="coerce"
    ).dropna().unique()), dtype="datetime64[ns]")
    quarter_ends = np.asarray(pd.to_datetime(
        [f"{year}-{month}-{day}" for year in range(2014, 2026)
         for month, day in (("03", "31"), ("06", "30"), ("09", "30"), ("12", "31"))],
        errors="coerce"), dtype="datetime64[ns]")

    summary = {"code": code, "n_days": int(len(k)), "n_pub": 0,
               "pub_covered": 0}
    all_events = []
    for metric, threshold in (("bps", BPS_STEP), ("eps", EPS_STEP)):
        indices = steps_of(k[metric], threshold)
        step_set = set(indices)
        matched_count = 0
        quarter_end_unmatched = 0
        values = k[metric].to_numpy(dtype=float)
        for index in indices:
            matched, matched_pub, gap = _nearest_publication(index, dates, publications)
            near_qe = _near_quarter_end(index, dates, quarter_ends)
            matched_count += int(matched)
            quarter_end_unmatched += int(near_qe and not matched)
            previous = values[index - 1]
            all_events.append({
                "code": code, "metric": metric, "date": dates.iloc[index],
                "relative_change": (values[index] - previous) / previous,
                "matched_publication": matched, "matched_pubDate": matched_pub,
                "publication_gap_trading_days": gap,
                "near_quarter_end": near_qe,
                "quarter_end_unmatched": bool(near_qe and not matched),
            })
        summary[f"{metric}_steps"] = len(indices)
        summary[f"{metric}_matched"] = matched_count
        summary[f"{metric}_unmatched"] = len(indices) - matched_count
        if metric == "bps":
            summary["qe_unmatched"] = quarter_end_unmatched
            effective = [pub for pub in publications
                         if dates.iloc[0] <= pd.Timestamp(pub) <= dates.iloc[-1]]
            summary["n_pub"] = len(effective)
            covered = 0
            for pub in effective:
                centre = int(dates.searchsorted(pd.Timestamp(pub)))
                lo = max(1, centre - TOL_HI)
                hi = min(len(k) - 1, centre + TOL_LO)
                covered += int(any(index in step_set for index in range(lo, hi + 1)))
            summary["pub_covered"] = covered
    return summary, pd.DataFrame(all_events)


def audit_one(code: str, start_year: int = 2015, refresh: bool = False):
    payload = cached_queries(code, start_year, refresh)
    k = pd.DataFrame(payload["k"], columns=["date", "close", "pbMRQ", "peTTM"])
    profit = pd.DataFrame(payload["profit"], columns=PROFIT_COLUMNS)
    return audit_frames(code, k, profit)


def select_sample(universe: pd.DataFrame, n: int = N_STOCKS, seed: int = SEED) -> pd.DataFrame:
    """Balanced exchange × listing-era × listing-status sample from the current universe."""
    data = universe.copy()
    data["ipoDate"] = pd.to_datetime(data["ipoDate"], errors="coerce")
    data = data[data["code"].astype(str).str.match(r"^(sh|sz)\.")]
    # Ensure at least roughly 400 trading observations are feasible before the sample end.
    data = data[data["ipoDate"] <= pd.Timestamp("2024-01-01")].copy()
    year = data["ipoDate"].dt.year
    data["listing_era"] = np.select([year < 2018, year <= 2020],
                                    ["pre2018", "2018_2020"], default="2021_2024")
    data["listing_status"] = np.where(data["status"].astype(int) == 1, "active", "ended")
    data["audit_stratum"] = (data["exchange"].astype(str) + "|" + data["listing_era"]
                             + "|" + data["listing_status"])
    rng = np.random.default_rng(seed)
    groups = []
    for stratum, group in sorted(data.groupby("audit_stratum"), key=lambda item: item[0]):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        groups.append((stratum, list(indices)))
    chosen = []
    while len(chosen) < min(n, len(data)) and any(indices for _, indices in groups):
        for _, indices in groups:
            if indices and len(chosen) < n:
                chosen.append(indices.pop())
    return data.loc[chosen].sort_values("audit_stratum").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_STOCKS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {login.error_msg}")
    try:
        universe = pd.read_parquet(CLEAN / "universe.parquet")
        picks = select_sample(universe, args.n, args.seed)
        rows, events, failures = [], [], []
        for i, record in enumerate(picks.to_dict("records"), 1):
            code = record["code"]
            log.info(f"[{i}/{len(picks)}] {code} | {record['audit_stratum']}")
            try:
                result, stock_events = audit_one(code, int(record["ipoDate"].year), args.refresh)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"{code} 审计失败：{exc}")
                failures.append({"code": code, "error": str(exc)})
                result, stock_events = None, pd.DataFrame()
            if result:
                for key in ("audit_stratum", "exchange", "listing_era", "listing_status", "ipoDate"):
                    result[key] = record[key]
                rows.append(result)
            if not stock_events.empty:
                events.append(stock_events)
            time.sleep(0.15)

        out = pd.DataFrame(rows)
        steps = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
        failure_frame = pd.DataFrame(failures, columns=["code", "error"])
        out.to_csv(REPORTS / "pit_audit_multistock.csv", index=False, encoding="utf-8-sig")
        steps.to_csv(REPORTS / "pit_audit_multistock_steps.csv", index=False, encoding="utf-8-sig")
        failure_frame.to_csv(REPORTS / "pit_audit_failures.csv", index=False, encoding="utf-8-sig")
        if out.empty:
            raise RuntimeError("No stock completed the PIT audit")

        totals = out[["bps_steps", "bps_matched", "bps_unmatched", "eps_steps",
                      "eps_matched", "eps_unmatched", "qe_unmatched"]].sum()
        coverage = out["pub_covered"].sum() / max(out["n_pub"].sum(), 1)
        strata = out.groupby("audit_stratum")["code"].agg(list)
        strata_text = "；".join(f"{name}: {len(codes)} 只" for name, codes in strata.items())
        incomplete = len(failure_frame) > 0 or len(out) < len(picks)
        md = [
            "# 多股票 PIT 审计（估值字段）", "",
            f"生成时间：{now_local():%Y-%m-%d %H:%M:%S}（北京时间）",
            f"样本：{len(out)}/{len(picks)} 只完成审计；从当前 universe.parquet 按交易所 × 上市年代 × 退市状态分层均衡抽取（seed={args.seed}）。",
            f"分层覆盖：{strata_text}。", "",
            "方法：implied BPS = close/pbMRQ、implied EPS = close/peTTM 在有效正值区间应近似分段常数。",
            f"台阶与公告日匹配使用实际交易行索引的 [−{TOL_LO}, +{TOL_HI}] 个交易日窗口；",
            f"季度末伪证使用季度末 ±{QE_TOL} 个交易日，而不是自然日近似。",
            "EPS 诊断仅覆盖 peTTM > 0.5 的观测，不能代表亏损公司的完整历史。", "",
            "| 指标 | 台阶数 | 匹配公告日 | 未匹配 |",
            "|---|---:|---:|---:|",
            f"| 隐含 BPS（pbMRQ） | {int(totals['bps_steps'])} | {int(totals['bps_matched'])} "
            f"({totals['bps_matched']/max(totals['bps_steps'],1):.1%}) | {int(totals['bps_unmatched'])} |",
            f"| 隐含 TTM EPS（peTTM） | {int(totals['eps_steps'])} | {int(totals['eps_matched'])} "
            f"({totals['eps_matched']/max(totals['eps_steps'],1):.1%}) | {int(totals['eps_unmatched'])} |",
            "",
            f"公告日 → BPS 台阶覆盖率：{out['pub_covered'].sum()}/{out['n_pub'].sum()}（{coverage:.1%}）。",
            f"未匹配且位于季度末 ±{QE_TOL} 个交易日的 BPS 台阶：{int(totals['qe_unmatched'])} "
            f"（占全部 BPS 台阶 {totals['qe_unmatched']/max(totals['bps_steps'],1):.1%}）。", "",
            "## 证据边界", "",
            "这些比例用于判断数据是否与按披露日更新相容，不能证明供应商从不修订历史，也不能替代全股票逐笔核验。",
            "公告日覆盖率还受阈值、公司行为和估值接近零影响，不应单独解释为数据错误率。",
            "所有台阶、匹配日期和季度末标记均保存在 pit_audit_multistock_steps.csv；未匹配事件未经过公司行动逐条外核前，不归因于送转、重组或补充公告。", "",
            f"查询完整性：{'不完整，存在失败记录，不能据此作最终判定' if incomplete else '所选样本查询均完成'}；失败明细见 pit_audit_failures.csv。",
        ]
        (REPORTS / "pit_audit_multistock.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        log.info("\n" + "\n".join(md[:18]))
        heartbeat("17 PIT 审计结束", log)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
