# Factor-crowding proxies in China's A-share market

This repository contains analysis code, configuration, and manuscript materials for a measurement comparison of return-correlation and margin-financing-balance proxies. The estimands are associational: neither proxy is interpreted as account-level total positions or as a direct observation of true crowding.

`paper/main.tex` is the current English manuscript and `paper/main.pdf` is its compiled version. `ARCHIVE_README.md` retains the Chinese archive notes. Earlier task plans are historical documents, not substitutes for the current manuscript.

## Rights and current data-release route

The code is under the MIT License in `LICENSE`. The manuscript and other original documents are copyright 2026 Jiaheng Wen and all rights are reserved for now. The MIT License does not cover manuscripts, third-party data, or provider-derived datasets.

The repository currently follows **route B** because asset-specific redistribution permission is unresolved. It does not distribute stand-alone raw, security-level, aggregate, or other provider-derived datasets. This exclusion includes the four aggregate inputs used in earlier local validation, independent result CSV/JSON files, raw provider data, daily/monthly panels, and complete calendars. Manuscript tables and figures remain part of the original manuscript, not downloadable datasets.

Raw daily market/valuation data come from Baostock and margin-financing data from public exchange disclosures. Researchers must obtain upstream inputs lawfully under the providers' terms and construct the analysis inputs themselves. The public package alone does not reproduce the empirical results. If written permission later covers specific assets, only those assets may be added, with matching documentation and reproduction claims.

## Environment

The verified core analysis environment is Python 3.12 on Windows. The fetch orchestrator is designed for Linux/macOS with Bash and `flock`; it is not a Windows one-click pipeline.

```bash
python -m venv .venv
```

- Windows PowerShell: `.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt`
- Linux/macOS: `.venv/bin/python -m pip install -r requirements-analysis.txt`

`requirements-analysis.txt` covers analysis, tests, tables, and figures. For upstream fetching, install `requirements.txt`; it adds pinned AkShare and Baostock versions. Installing dependencies does not validate upstream availability or historical-data completeness.

All analysis scripts locate the repository through `FIN_ROOT`:

- Windows PowerShell: `$env:FIN_ROOT = (Get-Location).Path`
- Linux/macOS: `export FIN_ROOT="$PWD"`

## What can be tested without provider-derived data

```bash
python -m unittest discover -s code/tests -p "test_*.py" -v
python code/25_margin_staleness_tests.py
```

The four second-moment integration tests explicitly skip when their three required aggregate inputs are absent. A skip is not an empirical validation. The remaining tests exercise synthetic or in-memory numerical and timing logic.

## Reproduction after independently constructing inputs

Earlier local route-A validation used four aggregate files under `data/clean/`: `factor_returns.parquet`, `crowding.parquet`, `comomentum_faithful.parquet`, and `factor_daily.parquet`. Those files are **not included** in route B. On 2026-10-24, a separate local package containing them passed 45 discovered tests plus four margin-staleness tests; ten analysis/table scripts completed and 29 declared outputs matched production byte-for-byte. That is local validation history, not a claim about this public package.

After lawfully obtaining upstream data and constructing the required inputs, researchers can inspect and run the analysis scripts directly. Scripts 09, 10, 13, 15, 16, 20, 21, 22, and 23 constitute the earlier bounded analysis refresh. Script 12 additionally requires independent summary CSVs not distributed here. Other scripts require some combination of security-level monthly/daily panels, calendar, raw margin files, or terminal-return intermediates. No end-to-end raw-data rebuild is claimed.

The terminal-return audit order is scripts 26, 27, 28, 29, then 12. Production panel construction retains historical and holding-period ST trading days; ST status is a formation-snapshot filter only. Script 29's 0% and −100% cases are stress scenarios, not estimated Chinese delisting, share-exchange, or merger payoffs.

## Full raw-data pipeline

On Linux/macOS, after installing `requirements.txt` and lawfully obtaining the raw inputs:

```bash
FIN_ROOT="$PWD" bash code/run_all.sh
```

The orchestrator includes scripts 01–30 and writes `data/run_manifest.json` only when that pipeline is actually run. Stage 12 must remain after the empirical scripts. Network access, provider schema changes, rate limits, and upstream terms can affect a fresh rebuild.

## Evidence boundaries

- Margin-balance changes show conditional negative associations with pooled future factor returns, concentrated in the low-side leg; results are not uniformly significant across factors or subsamples.
- Baseline daily comomentum has limited short-horizon associations. Weekly analogues produce exploratory long-horizon results that depend on construction and factor and are sensitive to multiple-testing corrections.
- Volatility, skewness, and bad-week outcomes are not mutually confirming and do not validate either proxy as true crowded positions.
- The main limitations include six factors, roughly 120 monthly observations, current-snapshot industries, and the institutional meaning of the margin proxy.
