#!/usr/bin/env bash
# ============================================================
# supervise.sh —— 守护进程
#
# 为什么需要它：流水线要连续跑几天，任何一次意外中断（进程被杀、网络抖动、
# 单点异常）都不应该让整夜的机器时间白费。run_all.sh 的每个阶段都是断点续传的，
# 所以「重启」是安全的、幂等的。
#
# 退出条件（任一满足即停）：
#   1. 存在 STOP 文件        → 人工叫停
#   2. 存在 data/DONE 文件   → 全部阶段跑完
#   3. 连续重启超过 MAX_RESTART 次 → 防止在硬故障上无限空转
# ============================================================
set -u
ROOT=/home/user/金融科研1014
CODE=$ROOT/code
LOGS=$ROOT/logs
MAX_RESTART=40
BACKOFF=20

mkdir -p "$LOGS"
cd "$CODE" || exit 1

restart=0
echo "[supervise] $(date -u '+%F %T') 守护进程启动，PID $$"
echo "[supervise] 退出条件：STOP 文件 / data/DONE / 连续重启超过 $MAX_RESTART 次"
while true; do
  if [ -f "$ROOT/STOP" ]; then
    echo "[supervise] $(date -u '+%F %T') 检测到 STOP 文件 → 退出守护"
    break
  fi
  if [ -f "$ROOT/data/DONE" ]; then
    echo "[supervise] $(date -u '+%F %T') 检测到 data/DONE → 全部阶段完成，退出守护"
    break
  fi

  bash "$CODE/run_all.sh" >> "$LOGS/runner.out" 2>&1
  rc=$?

  # rc=3 表示已有另一个实例持有锁 —— 不是故障，等它跑完即可，不计入重启次数
  if [ "$rc" -eq 3 ]; then
    echo "[supervise] $(date -u '+%F %T') 已有实例在运行，等待 60s 后重试"
    sleep 60
    continue
  fi

  if [ -f "$ROOT/STOP" ] || [ -f "$ROOT/data/DONE" ]; then
    echo "[supervise] $(date -u '+%F %T') run_all 退出 rc=$rc，且已满足退出条件"
    break
  fi

  restart=$((restart + 1))
  if [ "$restart" -gt "$MAX_RESTART" ]; then
    echo "[supervise] $(date -u '+%F %T') 已重启 $restart 次，超过上限 $MAX_RESTART → 停止守护，请人工检查"
    break
  fi
  echo "[supervise] $(date -u '+%F %T') run_all 退出 rc=$rc，第 $restart 次重启，${BACKOFF}s 后继续"
  sleep "$BACKOFF"
done
