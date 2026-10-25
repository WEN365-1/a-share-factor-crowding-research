"""
04_validate_roe.py —— 隐含 ROE 代理变量的抽样校验

背景（这是一个重要的设计简化）：
  原计划要从财务报表构建质量因子（ROE），需按披露日做 PIT 对齐，是全流程最重、最容易翻车的一块。
  但 Baostock 日频 k 线已经直接给出 peTTM 与 pbMRQ，而

        ROE = E/B = (E/P) / (B/P) = (1/peTTM) / (1/pbMRQ) = pbMRQ / peTTM

  隐含 ROE 因而可从日频估值字段推出，零额外拉取。多股票台阶审计支持其按披露日更新的
  解释，但不能保证供应商不修订历史，故不称其为“天然 PIT”。

本脚本的作用：随机抽 SAMPLE_N 只股票，拉真实季频 roeAvg（query_profit_data，带 pubDate），
与同期的隐含 ROE 对比，给出相关系数与分位偏差。若相关性足够高，代理变量即可写进论文；
若不高，回退到真实财报口径（则需另建全量财报层，工作量大幅上升）。

**对准口径（关键，容易做错）**：
  隐含 ROE 的观测窗口必须放在**披露日 pubDate 之后**，不能放在季度末之后。
  季度末之后、披露之前，市场价格里嵌的还是上一期的财报，拿它去比当季 ROE 是错位的，
  相关性会被无故压低。正确做法：看「披露之后市场有没有跳到新的隐含 ROE」。
  这样校验的也正是代理变量在论文里的实际用途——按当期已披露信息给股票排序。

**已知的口径差异（会压低相关系数，需在论文里说明）**：
  隐含 ROE = pbMRQ / peTTM 是 **TTM** 口径（TTM 净利润 / 最新报告期净资产），
  而 Baostock 的 `roeAvg` 是**单季平均**净资产收益率。
  两者定义不同，所以这里测的是「代理变量是否有信息量」，不是「是否相等」。
  对 QMJ 这种按截面排序使用的因子，有信息量就够了。

产出：reports/roe_proxy_validation.csv  +  reports/roe_proxy_validation.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import baostock as bs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN, DAILY, Gate, REPORTS, bs_login_ok, get_logger,  # noqa: E402
                    heartbeat, now_local, read_parquet, unit_path)

log = get_logger("04_roe", "04_validate_roe.log")

SAMPLE_N = 80          # 抽样股票数
POST_PUB_DAYS = 30     # 披露日之后的观测窗口长度
SEED = 20261016


def main() -> None:
    gate = Gate(log)
    uni = read_parquet(CLEAN / "universe.parquet")
    rng = np.random.default_rng(SEED)
    codes = sorted(rng.choice(uni["code"].values, size=min(SAMPLE_N, len(uni)), replace=False).tolist())
    log.info(f"抽样 {len(codes)} 只股票做隐含 ROE 校验 | 北京时间 {now_local():%Y-%m-%d %H:%M:%S}")

    if not bs_login_ok(bs, log):
        return

    rows = []
    for i, code in enumerate(codes):
        if not gate.ok():
            log.warning("闸门触发，提前收手")
            break
        p = unit_path(DAILY / "3", code)
        if not p.exists():
            continue
        d = read_parquet(p)
        if d.empty:
            continue
        d["date"] = pd.to_datetime(d["date"])
        for c in ("peTTM", "pbMRQ"):
            d[c] = pd.to_numeric(d[c], errors="coerce")

        # 先取真实 ROE 及其披露日
        real = []
        for year in range(2015, 2026):
            for q in (1, 2, 3, 4):
                try:
                    rs = bs.query_profit_data(code=code, year=year, quarter=q)
                    while rs.error_code == "0" and rs.next():
                        f = dict(zip(rs.fields, rs.get_row_data()))
                        if f.get("roeAvg") not in ("", None) and f.get("pubDate"):
                            real.append({"statDate": f["statDate"],
                                         "real_roe": float(f["roeAvg"]),
                                         "pubDate": pd.Timestamp(f["pubDate"])})
                except Exception:  # noqa: BLE001
                    continue

        # 隐含 ROE：只在**披露日之后**的窗口内取中位数（见文件头注释）
        for r in real:
            pub = r["pubDate"]
            win = d[(d["date"] > pub) & (d["date"] <= pub + pd.Timedelta(days=POST_PUB_DAYS))]
            if len(win) < 5:
                continue
            pe = win["peTTM"].median()
            pb = win["pbMRQ"].median()
            if pd.isna(pe) or pd.isna(pb) or pe <= 0 or pb <= 0:
                continue
            rows.append({"code": code, "statDate": r["statDate"], "pubDate": pub,
                         "implied_roe": pb / pe, "real_roe": r["real_roe"]})

        if (i + 1) % 10 == 0:
            log.info(f"已处理 {i + 1}/{len(codes)}")
    bs.logout()

    df = pd.DataFrame(rows)
    if df.empty:
        log.error("未取到数据（可能是 pubDate 字段缺失或日线文件尚未抓完）")
        return
    df.to_csv(REPORTS / "roe_proxy_validation.csv", index=False, encoding="utf-8-sig")

    # 横截面相关是这里的首要指标：QMJ 是按截面排序使用的
    piv = df.dropna(subset=["implied_roe", "real_roe"])
    if len(piv) < 20:
        log.error(f"可比观测只有 {len(piv)} 对，不足以评估")
        return
    corr = piv["implied_roe"].corr(piv["real_roe"])
    spear = piv["implied_roe"].corr(piv["real_roe"], method="spearman")
    # 再报一个逐股票内的时序相关，作为补充（横截面才是因子用途）
    within = (piv.groupby("code")
              .apply(lambda g: g["implied_roe"].corr(g["real_roe"]) if len(g) >= 6 else np.nan,
                     include_groups=False)
              .dropna())

    md = ["# 隐含 ROE 代理变量校验", "",
          f"生成时间：{now_local():%Y-%m-%d %H:%M:%S}（北京时间）", "",
          f"- 样本：{piv['code'].nunique()} 只股票，{len(piv)} 个 (股票, 报告期) 观测对",
          f"- 观测窗口：**披露日 pubDate 之后 {POST_PUB_DAYS} 天**内 peTTM / pbMRQ 的中位数",
          f"- Pearson 相关系数：**{corr:.3f}**",
          f"- Spearman 秩相关：**{spear:.3f}**",
          f"- 股票内时序相关系数（{len(within)} 只股票）：中位 {within.median():.3f}",
          "- 隐含 ROE 定义：`pbMRQ / peTTM`（TTM 口径，由日频估值字段推出；披露日更新有多股票样本证据）",
          "- 真实 ROE 定义：Baostock `query_profit_data` 的 `roeAvg`（单季平均口径，按 pubDate 对齐）",
          "",
          "## 怎么读", "",
          "两者口径并不相同（TTM vs 单季平均），所以这里检验的是**代理变量是否有信息量**，",
          "不是**是否相等**。QMJ 按截面排序使用，秩相关（Spearman）比 Pearson 更贴近实际用途。",
          "秩相关足够高即可放心使用代理变量；若偏低，则需回退到自建全量财报层",
          "（按披露日 PIT 对齐，约 23.8 万次请求），工作量与风险都大幅上升。"]
    (REPORTS / "roe_proxy_validation.md").write_text("\n".join(md), encoding="utf-8")
    log.info(f"隐含 ROE vs 真实 ROE：{len(piv)} 对 | Pearson {corr:.3f} | Spearman {spear:.3f}")
    heartbeat("04 隐含 ROE 校验结束", log)


if __name__ == "__main__":
    main()
