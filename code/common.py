"""
common.py —— 全流水线共用的工具：路径、日志、时间闸门、断点续传、限流重试。

设计要点（这是让流水线能无人值守跑整夜的关键）：
1. 每个最小工作单元（一只股票 / 一个交易日）产出独立文件，存在即跳过 → 断点续传
2. 时间闸门 + STOP 文件 → 到点或收到开关就优雅收手，不丢已完成的进度
3. 统一的心跳日志 → 早上读日志就知道跑到哪、还剩多久
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
# 支持用环境变量 FIN_ROOT 覆盖根目录，使全部脚本可以针对合成数据做冒烟测试，
# 而不触碰生产数据。默认仍是真实根目录。
ROOT = Path(os.environ.get("FIN_ROOT", "/home/user/金融科研1014"))
CODE = ROOT / "code"
DATA = ROOT / "data"
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"

RAW = DATA / "raw"
CLEAN = DATA / "clean"
DAILY = RAW / "daily"
MARGIN = RAW / "margin"
META = RAW / "meta"

STOP_FILE = ROOT / "STOP"          # touch 这个文件 = 全部阶段优雅停止
CONTROL_FILE = DATA / "CONTROL.json"  # {"paused_stages": ["07_build_factors"]}


def ensure_dirs() -> None:
    for p in (DATA, CLEAN, RAW, DAILY, MARGIN, META, LOGS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


def config_path(name: str) -> Path:
    """优先读取当前项目配置，允许 FIN_ROOT 指向独立 fixture。"""
    candidates = [CODE / name, ROOT / "code" / name, Path(__file__).resolve().parent / name]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到配置文件 {name}: {candidates}")


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config() -> dict:
    with config_path("config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()


# ------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------
def get_logger(name: str, logfile: str | None = None) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        fh = logging.FileHandler(LOGS / logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # 主日志：所有阶段汇总到一处，方便早上一次性读完
    fh_all = logging.FileHandler(LOGS / "pipeline.log", encoding="utf-8")
    fh_all.setFormatter(fmt)
    logger.addHandler(fh_all)
    return logger


# ------------------------------------------------------------------
# 时钟 / 闸门
# ------------------------------------------------------------------
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def now_local() -> dt.datetime:
    off = CFG["runner"].get("tz_offset_hours", 8)
    return now_utc() + dt.timedelta(hours=off)


def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Gate:
    """时间闸门 + 全局停止开关。每个工作单元开工前问一次 gate.ok()。"""

    def __init__(self, logger: logging.Logger | None = None):
        self.log = logger
        stop_at = CFG["runner"].get("stop_at")
        self.stop_at = None
        if stop_at:
            self.stop_at = dt.datetime.fromisoformat(str(stop_at))
        self.stage_start = time.time()
        self.max_stage_seconds = CFG["runner"].get("max_stage_hours", 12) * 3600

    def ok(self) -> bool:
        if STOP_FILE.exists():
            self._say_once("检测到 STOP 文件 → 优雅停止（已完成进度全部保留）")
            return False
        if self.stop_at and now_local() >= self.stop_at:
            self._say_once(f"到达闸门时间 {self.stop_at}（北京时间）→ 不再开新工作单元")
            return False
        if time.time() - self.stage_start > self.max_stage_seconds:
            self._say_once("本阶段达到 max_stage_hours → 收手，下一阶段可续跑")
            return False
        return True

    _said: set = set()

    def _say_once(self, msg: str) -> None:
        if self.log and msg not in self._said:
            self.log.warning(msg)
            self._said.add(msg)


def stage_allowed(stage: str) -> bool:
    """允许在运行中暂停某个阶段：data/CONTROL.json -> {"paused_stages": [...]}"""
    try:
        with open(CONTROL_FILE, "r", encoding="utf-8") as f:
            ctl = json.load(f)
    except Exception:
        return True
    if ctl.get("stop_all"):
        return False
    return stage not in set(ctl.get("paused_stages", []))


# ------------------------------------------------------------------
# 断点续传的朴素实现：单元级文件缓存
# ------------------------------------------------------------------
def unit_path(base: Path, key: str, suffix: str = ".parquet") -> Path:
    d = base / key[:6] if len(key) > 6 else base
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}{suffix}"


def done(base: Path, key: str) -> bool:
    p = unit_path(base, key)
    return p.exists() and p.stat().st_size > 0


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_all(base: Path, pattern: str = "*.parquet") -> pd.DataFrame:
    files = sorted(base.rglob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


# ------------------------------------------------------------------
# 带退避重试的请求包装
# ------------------------------------------------------------------
def with_retry(fn, logger: logging.Logger, what: str, retry: int = 5, base_sleep: float = 2.0):
    last = None
    for attempt in range(retry):
        try:
            out = fn()
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            wait = base_sleep * (2 ** attempt)
            logger.warning(f"{what} 第 {attempt + 1}/{retry} 次失败：{type(e).__name__}: {e}；{wait:.0f}s 后重试")
            time.sleep(wait)
    logger.error(f"{what} 重试 {retry} 次仍失败，跳过。最后错误：{last}")
    return None


# ------------------------------------------------------------------
# 进度报告器
# ------------------------------------------------------------------
class Progress:
    def __init__(self, total: int, logger: logging.Logger, name: str, every: int = 50):
        self.total, self.log, self.name, self.every = total, logger, name, every
        self.start = time.time()
        self.n = 0
        self.skipped = 0

    def tick(self, skipped: bool = False) -> None:
        self.n += 1
        if skipped:
            self.skipped += 1
        if self.n % self.every == 0 or self.n == self.total:
            el = time.time() - self.start
            rate = el / max(self.n - self.skipped, 1)
            remain = rate * (self.total - self.n)
            self.log.info(
                f"[{self.name}] {self.n}/{self.total} "
                f"(跳过 {self.skipped}) 用时 {fmt_eta(el)} 预计剩余 {fmt_eta(remain)}"
            )

    def done(self) -> None:
        el = time.time() - self.start
        self.log.info(f"[{self.name}] 完成 {self.n}/{self.total}，其中跳过 {self.skipped}，总用时 {fmt_eta(el)}")


def heartbeat(msg: str, logger: logging.Logger) -> None:
    logger.info(f"♥ {msg} | 北京时间 {now_local():%Y-%m-%d %H:%M:%S} | UTC {now_utc():%Y-%m-%d %H:%M:%S}")


def bs_login_ok(bs, log, tries: int = 30, base_wait: float = 5.0) -> bool:
    """
    带退避的 Baostock 登录。

    为什么必须重试：Baostock 的登录会偶发返回「网络接收错误」。原来的写法是一次失败就
    `sys.exit(1)`，整个分片直接白跑——已实测：日线分片 5 因此丢掉 678 只股票，
    而且因为进程是"正常退出"，runner 只看到 rc=1，没有任何显眼的失败信号。
    宁可在这里等，也不要让一个分片悄无声息地少一批数据。
    """
    for k in range(tries):
        msg = "?"
        try:
            lg = bs.login()
            if getattr(lg, "error_code", "1") == "0":
                return True
            msg = getattr(lg, "error_msg", "?")
        except Exception as e:  # noqa: BLE001
            msg = repr(e)
        wait = min(base_wait * (k + 1), 60.0)
        log.warning(f"Baostock 登录失败（{msg}），{wait:.0f}s 后重试（第 {k + 1}/{tries} 次）")
        time.sleep(wait)
    log.error(f"Baostock 登录连续 {tries} 次失败，放弃")
    return False


# ---------------------------------------------------------------
# 形成月股票池、回归排序与推断工具
# ---------------------------------------------------------------

def formation_universe(frame: pd.DataFrame, max_nontrading_days: int = 10) -> pd.DataFrame:
    """按形成日可见信息筛选股票，不删除历史或持有期 ST 收益。

    ST 状态取自当月最后可见交易快照；缺该字段时直接失败，避免静默退回
    旧的日线删除口径。非交易日分母使用当月交易日历天数，而非固定观测数。
    """
    required = {"float_mv", "n_days", "calendar_days", "isST"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"形成月股票池缺少字段: {sorted(missing)}")
    st = pd.to_numeric(frame["isST"], errors="coerce")
    return frame[(frame["float_mv"].notna()) & (frame["float_mv"] > 0)
                 & st.eq(0) & st.notna()
                 & ((frame["calendar_days"] - frame["n_days"]) <= max_nontrading_days)].copy()


def realize_leg_return(leg: pd.DataFrame, formation_w: pd.Series) -> tuple[float, float, int]:
    """冻结形成月权重，在次月收益非缺失者上重新归一化并报告覆盖率。"""
    valid = leg["ret_next"].notna()
    coverage = float(formation_w[valid].sum())
    if not valid.any() or coverage <= 0:
        return np.nan, coverage, 0
    w_valid = formation_w[valid] / coverage
    ret = float((leg.loc[valid, "ret_next"] * w_valid).sum())
    return ret, coverage, int(valid.sum())


def regression_frame(frame: pd.DataFrame, z_by: str = "factor") -> pd.DataFrame:
    """观测回归与置换回归共用同一行顺序：先时间、再分组。"""
    return frame.sort_values(["ym", z_by]).copy()


def time_index(values) -> np.ndarray:
    """把 ym（Period 序列 / PeriodIndex）转成整数月序号，供 Driscoll–Kraay 的 time 参数。"""
    return np.asarray(pd.PeriodIndex(values, freq="M").asi8, dtype=np.int64)


def driscoll_kraay_cov(X, resid, time, lags: int) -> np.ndarray:
    """Driscoll–Kraay 协方差（Bartlett 核）。

    面板是「因子 × 月」堆叠时，普通 Newey–West 的滞后按**行**算，6 行只覆盖约
    一个月，还会把同月不同因子之间的协方差混进来。DK 的做法是先把每个月内所有
    观测的得分 x_it·e_it 加总成月度向量 S_t，再对 S_t 做滞后 L 个**月**的 NW。
    缺失月份不补零，按实际出现的月份顺序排列。

    参数里的 X 必须已含常数项且与 resid 行数一致。
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(time).ravel()
    if t.shape[0] != X.shape[0]:
        raise ValueError(f"time 长度 {t.shape[0]} 与回归样本 {X.shape[0]} 不一致")
    u = X * np.asarray(resid, dtype=float)[:, None]
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    u_sorted = u[order]
    _, starts = np.unique(t_sorted, return_index=True)
    S = np.add.reduceat(u_sorted, starts, axis=0)      # (T, k)：每月得分之和
    omega = S.T @ S
    for lag in range(1, int(lags) + 1):
        if lag >= S.shape[0]:
            break
        w = 1.0 - lag / (lags + 1.0)
        G = S[lag:].T @ S[:-lag]
        omega += w * (G + G.T)
    # 用伪逆而不是 inv：剔除某个因子后的样本只剩 5 个因子，5 个固定效应虚拟变量
    # 加常数项恰好共线，X'X 奇异。statsmodels 的 HAC 本来就用 pinv，这里保持一致，
    # 否则会在「leave one factor out」这类样本上把可识别的测度系数变成 NaN。
    A = np.linalg.pinv(X.T @ X)
    return A @ omega @ A


def hac_ols(y, X, names: list[str], lags: int = 6, time=None):
    """普通最小二乘 + HAC 标准误。返回 {变量名: {coef, se, t, p}}。

    ``time=None``：对单序列做普通 Newey–West（与旧行为完全一致）。
    ``time`` 给定：对堆叠面板做 Driscoll–Kraay，``time`` 是每行的整数月序号；
    滞后 ``lags`` 因此按**月**而不是按行计算。

    会先丢掉**零方差列**：当某个因子在当期样本里整体缺失（例如 MOM 需要 12 个月
    完整历史、样本前段为空），它的固定效应虚拟变量会退化成全零列，与常数项完全共线。
    statsmodels 此时只发一条 SingularMatrixWarning 并把系数算成不可识别的值，
    不丢列的话回归结果会静默地变成垃圾。
    """
    import statsmodels.api as sm
    from scipy.stats import norm
    X = np.asarray(X, dtype=float)
    names = list(names)
    dropped: list[str] = []
    if X.ndim == 2 and X.shape[1]:
        sd = X.std(axis=0)
        keep = np.isfinite(sd) & (sd > 0)
        if not keep.all():
            dropped = [n for n, k in zip(names, keep) if not k]
            X = X[:, keep]
            names = [n for n, k in zip(names, keep) if k]
    Xc = sm.add_constant(X, has_constant="add")
    if time is None:
        res = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        params, bse, tvals, pvals = res.params, res.bse, res.tvalues, res.pvalues
    else:
        res = sm.OLS(y, Xc).fit()
        params = res.params
        var = np.diag(driscoll_kraay_cov(Xc, res.resid, time, lags))
        # 数值上可能出现极小的负方差（对称化误差），截到 0 再开方。
        bse = np.sqrt(np.clip(var, 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = np.asarray(params) / bse
        pvals = 2.0 * norm.sf(np.abs(tvals))
    out = {}
    for n, b, se, t, p in zip(["const"] + names, params, bse, tvals, pvals):
        out[n] = {"coef": float(b), "se": float(se), "t": float(t), "p": float(p)}
    out["_n"] = int(res.nobs)
    out["_r2"] = float(res.rsquared)
    out["_dropped"] = dropped
    return out


def _grouped_circular_shift(frame: pd.DataFrame, measure: str, z_by: str,
                             shift: int) -> pd.DataFrame:
    """按分组循环平移测度，返回供同一 estimator 重估的完整面板。"""
    out = frame.copy()
    # 对每个组按时间排序后平移；调用方提供的 frame 已包含完整设计矩阵。
    for _, idx in frame.groupby(z_by, observed=True, sort=False).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        ordered = frame.loc[idx].sort_values("ym").index
        values = frame.loc[ordered, measure].to_numpy(dtype=float)
        out.loc[ordered, measure] = np.roll(values, shift % len(values))
    return out


def placebo_null_dist(d, cols: list[str], measure: str, y: str,
                      z_by: str = "factor", hac_lags: int = 6, use_time: bool = True):
    """与观测回归完全相同 estimator 的 pooled |t| 零分布。

    每个 circular shift 都复制完整面板、在各分组内平移测度，并调用同一个
    ``hac_ols``。因此固定效应、控制变量、有效样本和 HAC 计算方式与观测
    statistic 一致；不会再把 factor-specific t 值混合成 pooled null。

    ``use_time=True`` 时按 Driscoll–Kraay 计算（堆叠面板）；单因子时间序列
    （每个 ym 只有一行）的调用者应传 ``use_time=False``，退回普通 Newey–West。
    """
    frame = d.sort_values(["ym", z_by]).copy()
    groups = [g for _, g in frame.groupby(z_by, observed=True, sort=False)]
    if not groups or any(len(g) < 30 for g in groups):
        return np.asarray([], dtype=float)
    # 只使用所有分组都支持的位移，避免较短分组被 modulo 重复使用同一位移。
    n_shifts = min(len(g) for g in groups) - 1
    # 平移不改变行顺序或月份，月序号在循环外算一次即可。
    tvals = time_index(frame["ym"]) if use_time else None
    ts = []
    for k in range(1, n_shifts + 1):
        perm = _grouped_circular_shift(frame, measure, z_by, k)
        r = hac_ols(perm[y].to_numpy(dtype=float), perm[cols].to_numpy(dtype=float), cols,
                    lags=hac_lags, time=tvals)
        t = r.get(measure, {}).get("t", np.nan)
        if np.isfinite(t):
            ts.append(abs(float(t)))
    return np.asarray(ts, dtype=float)


def placebo_pvalue(d, cols: list[str], measure: str, y: str,
                   obs_t: float, z_by: str = "factor", hac_lags: int = 6,
                   use_time: bool = True) -> tuple[float, int]:
    """计算有限循环置换 p 值，并使用加一修正。"""
    a = placebo_null_dist(d, cols, measure, y, z_by, hac_lags=hac_lags, use_time=use_time)
    if not a.size or not np.isfinite(obs_t):
        return float("nan"), int(a.size)
    return float((1.0 + np.sum(a >= abs(obs_t))) / (1.0 + a.size)), int(a.size)
