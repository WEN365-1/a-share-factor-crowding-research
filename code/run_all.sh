#!/usr/bin/env bash
# ============================================================
# run_all.sh —— 全流水线编排器
#
# 特点：
#   1. 断点续传：每个阶段自己跳过已完成的工作单元，重跑无害
#   2. 优雅停止：任何时刻 `touch /home/user/金融科研1014/STOP` 即停止
#   3. 分阶段暂停：编辑 data/CONTROL.json 的 paused_stages 可单独跳过某阶段
#   4. 全程日志：logs/pipeline.log（汇总）+ logs/<阶段>.log（分阶段）+ logs/runner.out
#
# 用法：
#   nohup bash run_all.sh > ../logs/runner.out 2>&1 &
#   tail -f ../logs/pipeline.log
# ============================================================
set -Eeuo pipefail
ROOT=${FIN_ROOT:-/home/user/金融科研1014}
CODE=$ROOT/code
LOGS=$ROOT/logs
MANIFEST=$ROOT/data/run_manifest.json
mkdir -p "$LOGS" "$ROOT/data"
cd "$CODE" || exit 1

# ---------- 互斥锁：防止出现两份实例争抢同一批文件写入 ----------
exec 201>"$ROOT/data/.pipeline.lock"
if ! flock -n 201; then
  echo ">>> 已有另一个 run_all 实例在运行（锁被占用），本实例退出 rc=3"
  exit 3
fi

# ---------- 运行 manifest：记录阶段、时间和返回码，避免结果版本漂移 ----------
MANIFEST="$ROOT/data/run_manifest.json"
manifest_start () {
  python3 - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"started_utc": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), "stages": []}, ensure_ascii=False, indent=2) + "\n")
PY
}
manifest_stage () {
  local script="$1"; local rc="$2"
  python3 - "$MANIFEST" "$script" "$rc" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1])
data=json.loads(p.read_text()) if p.exists() else {"stages": []}
data.setdefault("stages", []).append({"script": sys.argv[2], "rc": int(sys.argv[3]), "finished_utc": datetime.now(timezone.utc).isoformat()})
p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
PY
}
manifest_start
if ! flock -n 201; then
  echo ">>> 已有另一个 run_all 实例在运行（锁被占用），本实例退出 rc=3"
  exit 3
fi

banner () { echo; echo "############ $1  $(date -u '+%F %T') UTC ############"; }

run_stage () {
  local script="$1"
  if [ -f "$ROOT/STOP" ]; then
    echo ">>> STOP 文件存在 → 停止流水线"
    return 2
  fi
  banner "START $script"
  python3 "$script"
  local rc=$?
  banner "END   $script (rc=$rc)"
  if [ "$rc" -ne 0 ]; then
    manifest_stage "$script" "$rc"
    echo ">>> 阶段 $script 失败，停止后续阶段"
    return "$rc"
  fi
  manifest_stage "$script" "$rc"
  return 0
}

run_stage_or_stop () {
  run_stage "$1" || exit $?
}

echo "============================================================"
echo "流水线启动  $(date -u '+%F %T') UTC  |  北京时间 $(TZ=Asia/Shanghai date '+%F %T')"
echo "============================================================"

# ---------- 阶段 1：数据抓取（两个数据源互不干扰，并行跑以压缩墙钟时间）----------
if [ -f "$ROOT/data/clean/universe.parquet" ] && [ -f "$ROOT/data/clean/calendar.parquet" ]; then
  echo ">>> 阶段 01 产出已存在，跳过"
else
  run_stage_or_stop 01_fetch_calendar.py
fi

N_SHARDS=${N_SHARDS:-8}

# 分片会因为偶发网络错误中途退出。实测发生过一次：日线分片 5 的 Baostock 登录返回
# "网络接收错误"，脚本静默 sys.exit，丢掉 678 只股票，而主流程毫无察觉地继续往下跑，
# 05 会在一份不完整的面板上装配。所以改成「跑一轮 → 校验 → 不齐就再来一轮」。
# 所有抓取都可续传（已存在的文件直接跳过），重跑很便宜，代价远小于带着残缺数据跑完全程。
MAX_FETCH_ROUNDS=4
for ROUND in $(seq 1 $MAX_FETCH_ROUNDS); do
  banner "抓取第 ${ROUND}/${MAX_FETCH_ROUNDS} 轮：日线 ${N_SHARDS} 分片 ∥ 两融明细"
  PIDS=""
  for i in $(seq 0 $((N_SHARDS - 1))); do
    python3 02_fetch_daily.py --shard "$i" --nshards "$N_SHARDS" &
    PIDS="$PIDS $!"
  done
  python3 03_fetch_margin.py &
  PIDS="$PIDS $!"
  for pid in $PIDS; do
    wait "$pid" || true        # 单个分片失败不算整体失败，由下面的校验决定是否再来一轮
  done
  if python3 check_fetch.py; then
    echo ">>> 抓取完整，进入清洗阶段"
    break
  fi
  if [ "$ROUND" = "$MAX_FETCH_ROUNDS" ]; then
    manifest_stage "fetch_integrity" 1
    echo ">>> 警告：跑满 ${MAX_FETCH_ROUNDS} 轮仍不完整，停止流水线。"
    exit 1
  else
    echo ">>> 第 ${ROUND} 轮后仍不完整，30s 后重试缺失部分"
    sleep 30
  fi
done

# ---------- 阶段 3：清洗与面板装配 ----------
run_stage_or_stop 05_build_panel.py

# ---------- 阶段 4：因子层 ----------
run_stage_or_stop 06_build_factors.py

# ---------- 阶段 5：因子评估层（因子看板）----------
run_stage_or_stop 07_evaluate.py

# ---------- 阶段 6：两个拥挤度测度与周频 LP 对照 ----------
run_stage_or_stop 08_crowding.py
run_stage_or_stop 19_comomentum_faithful.py
run_stage_or_stop 20_comomentum_faithful_tests.py

# ---------- 阶段 7：主检验 + 三层正交化 ----------
run_stage_or_stop 09_regressions.py

# ---------- 阶段 8：尾部检验 ----------
run_stage_or_stop 10_tail_risk.py

# ---------- 阶段 9：稳健性 ----------
run_stage_or_stop 11_robustness.py

# ---------- 阶段 10：检验功效 / 最小可检测效应 ----------
# 放在 11 之后：它复用 09 的面板构造，且要等拥挤度表稳定下来
run_stage_or_stop 13_power.py

# ---------- 阶段 11：实现波动、偏度与坏周诊断 ----------
# 这些诊断描述不同风险结果，不能用来证明测度实现正确或验证真实拥挤。
run_stage_or_stop 15_volatility.py

# ---------- 阶段 12：扩展、流量、周频对照、二阶矩与阈值敏感性 ----------
run_stage_or_stop 16_extension_analyses.py
run_stage_or_stop 17_pit_audit.py
run_stage_or_stop 18_flow_proxy.py
run_stage_or_stop 21_comomentum_robustness.py
run_stage_or_stop 22_comomentum_factor_slopes.py
run_stage_or_stop 23_lp_second_moments.py
run_stage_or_stop 24_margin_staleness_robustness.py
run_stage_or_stop 25_margin_staleness_tests.py

# ---------- 阶段 13：形成窗口、收益实现与终止事件审计 ----------
# 26/27 使用派生面板；28 还需要未按 ST 过滤的原始日线和 universe。
# 29 的 0%/-100% 仅为终止收益压力情景，不是中国退市或换股 payoff 估计。
run_stage_or_stop 26_formation_timing_audit.py
run_stage_or_stop 27_formation_robustness.py
run_stage_or_stop 28_terminal_return_audit.py
run_stage_or_stop 29_terminal_return_robustness.py
# 30 在全部生产输入固定后审计 comomentum 股票抽样；不替换生产测度。
run_stage_or_stop 30_comomentum_sampling.py

# ---------- 阶段 14：论文图表（CSV → LaTeX）----------
# 必须在全部实证脚本之后：它是纯展示层，读取它们的输出。
run_stage_or_stop 12_make_tables.py

# ---------- 阶段 15：数据质量审计 ----------
run_stage_or_stop 14_audit_data.py

# ---------- 阶段 16：隐含 ROE 代理变量校验（要打网络，放最后）----------
run_stage_or_stop 04_validate_roe.py

echo
echo "============================================================"
echo "流水线走完  $(date -u '+%F %T') UTC  |  北京时间 $(TZ=Asia/Shanghai date '+%F %T')"
echo "============================================================"
date -u '+%F %T' > "$ROOT/data/DONE"
