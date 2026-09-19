#!/usr/bin/env bash
# ============================================================
# run_all.sh — 一键运行：统计 → ML → DL → 最终对比
# 用法：在项目根目录（本文件所在目录）执行  ./run_all.sh
#   - 默认跑 统计 + ML + 最终对比（stat vs ml）
#   - 加 --with-dl 参数，额外跑 DL（N-BEATS，CPU 下较慢）
# 脚本会自动：
#   1) 依次跑 统计/ML/DL 三个项目，生成各自的 2021 预测
#   2) 自动定位统计项目"最新 run"的 test_set_predictions.csv
#   3) 跑最终对比，输出 SMAPE 对比表
# ============================================================
set -euo pipefail

WITH_DL=0
if [[ "${1:-}" == "--with-dl" ]]; then
  WITH_DL=1
fi

# 定位脚本所在目录（即项目根）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=============================================="
echo " 项目根目录: $ROOT"
echo "=============================================="

# ---------- 0. 环境检查 ----------
for pycmd in python python3; do
  if command -v "$pycmd" >/dev/null 2>&1; then PY="$pycmd"; break; fi
done
if [[ -z "${PY:-}" ]]; then echo "[错误] 未找到 python，请先安装"; exit 1; fi
echo "使用解释器: $PY"

# ---------- 1. 统计项目 ----------
echo ""
echo "[1/4] 统计项目（time-series-forecast）..."
cd "$ROOT/time-series-forecast"
"$PY" main.py --exp exp_baseline
# 定位最新 run 的 test_set_predictions.csv
STAT_PRED="$(ls -td runs/*/test_set_predictions.csv 2>/dev/null | head -1 || true)"
if [[ -z "$STAT_PRED" ]]; then
  echo "[错误] 统计项目未生成 test_set_predictions.csv"; exit 1
fi
STAT_PRED="$(cd "$ROOT/time-series-forecast" && pwd)/$STAT_PRED"
echo "  统计侧 2021 预测: $STAT_PRED"

# ---------- 2. ML 项目 ----------
echo ""
echo "[2/4] ML 项目（ml_project）..."
cd "$ROOT/ml_project"
"$PY" main.py
ML_PRED="$ROOT/ml_project/output/ml_test_predictions.csv"
[[ -f "$ML_PRED" ]] || { echo "[错误] ML 项目未生成 $ML_PRED"; exit 1; }
echo "  ML 2021 预测: $ML_PRED"

# ---------- 3. DL 项目（可选） ----------
DL_ARGS=()
if [[ "$WITH_DL" -eq 1 ]]; then
  echo ""
  echo "[3/4] DL 项目（dl_project, N-BEATS，耗时较长）..."
  cd "$ROOT/dl_project"
  "$PY" main.py
  DL_PRED="$ROOT/dl_project/output/dl_test_predictions.csv"
  [[ -f "$DL_PRED" ]] || { echo "[错误] DL 项目未生成 $DL_PRED"; exit 1; }
  DL_ARGS=(--dl "$DL_PRED")
  echo "  DL 2021 预测: $DL_PRED"
else
  echo ""
  echo "[3/4] 跳过 DL（如需深度时序对比，加 --with-dl 参数）"
fi

# ---------- 4. 最终对比 ----------
echo ""
echo "[4/4] 最终对比（final_compare）..."
TEST_CSV="$ROOT/time-series-forecast/data/test.csv"
cd "$ROOT/final_compare"
"$PY" main.py \
  --stat "$STAT_PRED" \
  --ml   "$ML_PRED" \
  "${DL_ARGS[@]:-}" \
  --test "$TEST_CSV"

echo ""
echo "=============================================="
echo " 全部完成！对比结果见 final_compare/output/"
echo "=============================================="
