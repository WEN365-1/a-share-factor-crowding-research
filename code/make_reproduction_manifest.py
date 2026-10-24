"""Record an actually executed derived-data analysis refresh and its file hashes.

This command records evidence; it does not run analyses and must not be used to claim
that a raw-data pipeline was executed. Pass only scripts/tests that actually ran.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(os.environ.get("FIN_ROOT", Path(__file__).resolve().parents[1]))
INPUTS = [
    "data/clean/factor_returns.parquet",
    "data/clean/crowding.parquet",
    "data/clean/comomentum_faithful.parquet",
    "data/clean/factor_daily.parquet",
    "data/clean/monthly.parquet",
    "data/clean/calendar.parquet",
    "data/clean/universe.parquet",
]
INPUT_COLLECTIONS = ["data/clean/panel"]
TERMINAL_OUTPUTS = [
    "reports/terminal_return_events.csv",
    "reports/terminal_factor_returns.csv",
    "reports/terminal_leg_exposure.csv",
    "reports/terminal_stock_factor_exposure.csv",
    "reports/terminal_return_audit.json",
    "reports/terminal_return_regressions.csv",
    "reports/terminal_return_robustness_summary.json",
]
CORE_OUTPUTS = [
    "reports/table2_main.csv",
    "reports/table3_orthogonalization.csv",
    "reports/table6_compound.csv",
    "reports/table8_legs.csv",
    "reports/table9_robust_extra.csv",
    "reports/table10_agreement.csv",
    "reports/table_cov_stats.csv",
    "reports/comomentum_faithful_regressions.csv",
    "reports/comomentum_sampling_monthly.csv",
    "reports/comomentum_sampling_regressions.csv",
    "reports/comomentum_sampling_correlations.csv",
    "reports/comomentum_sampling_summary.json",
]
FORMATION_OUTPUTS = [
    "reports/formation_timing_audit.json",
    "reports/formation_common_month_end_sensitivity.csv",
    "reports/formation_membership_timing.csv",
    "reports/formation_daily_compounding_sensitivity.csv",
    "reports/formation_daily_leg_sensitivity.csv",
    "reports/formation_robustness_regressions.csv",
    "reports/formation_robustness_summary.json",
    "paper/tables/table9_formation.tex",
]
OUTPUTS = CORE_OUTPUTS + FORMATION_OUTPUTS + TERMINAL_OUTPUTS
PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "pyarrow", "PyYAML", "matplotlib", "Pillow"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(root: Path, rel: str) -> dict:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    return {"path": rel, "bytes": path.stat().st_size, "sha256": digest(path)}


def collection_entry(root: Path, rel: str, files: list[Path] | None = None,
                     selection: str = "all parquet files recursively") -> dict:
    base = root / rel
    files = sorted(p for p in base.rglob("*.parquet") if p.is_file()) if files is None else sorted(files)
    if not files:
        raise FileNotFoundError(f"Required input collection is empty: {base}")
    combined = hashlib.sha256()
    total_bytes = 0
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Required collection input is missing: {path}")
        relative = path.relative_to(base).as_posix().encode("utf-8")
        file_hash = digest(path).encode("ascii")
        combined.update(len(relative).to_bytes(8, "big") + relative)
        combined.update(file_hash)
        total_bytes += path.stat().st_size
    return {"path": rel, "files": len(files), "bytes": total_bytes,
            "selection": selection,
            "combined_sha256": combined.hexdigest(),
            "method": "concatenate relative-path length, UTF-8 relative path, and file SHA-256 in sorted order"}


def raw_daily_collection(root: Path) -> dict:
    """Hash exactly the stock files read by script 28, not unrelated raw units."""
    codes = sorted(pq.read_table(root / "data/clean/monthly.parquet", columns=["code"])
                   .column("code").to_pandas().astype(str).unique())
    base = root / "data/raw/daily/3"
    files = [base / code[:6] / f"{code}.parquet" for code in codes]
    return collection_entry(root, "data/raw/daily/3", files,
                            "one file per unique code in data/clean/monthly.parquet; script 28 input set")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ran", action="append", required=True,
                        help="Code-relative script path that actually ran; repeat as needed.")
    parser.add_argument("--tests", action="append", required=True,
                        help="Human-readable test command/scope that actually passed; repeat as needed.")
    parser.add_argument("--reference-root", type=Path,
                        help="Optional reference tree for byte comparison of the eight outputs.")
    args = parser.parse_args()

    runs = []
    for script in args.ran:
        rel = f"code/{script}" if not script.startswith("code/") else script
        runs.append(entry(ROOT, rel))
    outputs = [entry(ROOT, rel) for rel in OUTPUTS]
    comparison = None
    if args.reference_root is not None:
        comparison = []
        for rel in CORE_OUTPUTS:
            item = entry(ROOT, rel)
            reference = entry(args.reference_root, rel)
            comparison.append({**item, "reference_sha256": reference["sha256"],
                               "identical_bytes": item["sha256"] == reference["sha256"]})
        if not all(x["identical_bytes"] for x in comparison):
            raise RuntimeError("At least one clean output differs from its reference output")

    manifest = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "derived-data analysis refresh, formation-return audit, and raw-daily terminal-return audit",
        "raw_provider_data_rebuilt": False,
        "full_raw_to_paper_pipeline_executed": False,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        },
        "scripts_actually_run": runs,
        "tests_actually_passed": args.tests,
        "inputs": [entry(ROOT, rel) for rel in INPUTS],
        "input_collections": ([collection_entry(ROOT, rel) for rel in INPUT_COLLECTIONS]
                              + [raw_daily_collection(ROOT)]),
        "outputs": outputs,
        "reference_output_comparison": comparison,
        "reference_comparison_scope": (
            "Only the original eight core outputs; formation and terminal-return scripts were not run in the archived clean environment"
            if comparison is not None else
            "No reference-tree byte comparison requested; current production results intentionally differ from the pre-switch reference"
        ),
        "limitations": [
            "The manifest records bounded derived-input analyses and an audit using existing raw daily files, not upstream retrieval or raw-data reconstruction.",
            "Network availability, provider schemas, and historical raw-data completeness require a separate full-pipeline audit.",
            "The common-month-end regression is a partial timing check; 0% and -100% terminal cases are stress scenarios, not estimated payoffs.",
        ],
    }
    target = ROOT / "data" / "reproduction_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
