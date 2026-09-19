# -*- coding: utf-8 -*-
"""
final_compare/main.py — 统计 vs 机器学习 vs 深度学习 最终对比（2021 测试集，一次性）

只读输入：
  1) stat 预测（统计项目 test_set_predictions.csv）：unique_id, ds, best_forecast
  2) ml  预测（ML 项目 ml_test_predictions.csv）   ：unique_id, ds, ml_forecast
  3) dl  预测（DL 项目 dl_test_predictions.csv）   ：unique_id, ds, dl_forecast（可选）
  4) 测试集真实值（2021）：unique_id, ds, y

一次性输出各模型在 2021 测试集上的 SMAPE 对比表。

⚠️ 决策规则（务必提前定好，勿据此回改模型）：
  本结果只用于「确认 + 展示」，不作为回头改模型的依据。
  - 是否 per-SKU 混合、选用哪个模型，应以训练集 CV 的模型分布为准，而非本测试集结果。
  - 一旦用本测试集结果反哺建模决策，测试集即失效。

用法：
  python main.py \
    --stat <统计>/<run_id>/test_set_predictions.csv \
    --ml   <ml>/output/ml_test_predictions.csv \
    --dl   <dl>/output/dl_test_predictions.csv \
    --test <数据>/new_test.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from utilsforecast.evaluation import evaluate as uf_evaluate
from utilsforecast.losses import smape as uf_smape


def _smape_global(actual, pred) -> float:
    """全局 SMAPE，与 utilsforecast.losses.smape 的逐点公式完全一致：
    mean(|p-y|/(|y|+|p|))，除零计 0，返回 0~1 比值（不乘 100）。"""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.abs(a) + np.abs(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.abs(p - a) / denom
    return float(np.where(denom == 0, 0.0, s).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", required=True, help="统计侧2021测试期逐日预测 csv")
    ap.add_argument("--ml", required=True, help="ML侧2021测试期逐日预测 csv")
    ap.add_argument("--dl", default=None, help="DL(N-BEATS)侧2021测试期逐日预测 csv（可选）")
    ap.add_argument("--test", required=True, help="测试集真实值 csv（含 unique_id, ds, y）")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "output" / "final_compare.csv"))
    args = ap.parse_args()

    stat = pd.read_csv(args.stat, parse_dates=["ds"])
    ml = pd.read_csv(args.ml, parse_dates=["ds"])
    test = pd.read_csv(args.test, parse_dates=["ds"])

    if "best_forecast" not in stat.columns:
        raise ValueError("stat 预测需含 best_forecast 列（统计项目 test_set_predictions.csv）")
    if "ml_forecast" not in ml.columns:
        raise ValueError("ml 预测需含 ml_forecast 列（ML 项目 ml_test_predictions.csv）")

    stat = stat[["unique_id", "ds", "best_forecast"]].rename(columns={"best_forecast": "stat"})
    ml = ml[["unique_id", "ds", "ml_forecast"]].rename(columns={"ml_forecast": "ml"})
    y = test[["unique_id", "ds", "y"]]

    merged = y.merge(stat, on=["unique_id", "ds"], how="inner") \
              .merge(ml, on=["unique_id", "ds"], how="inner")

    models = ["stat", "ml"]
    if args.dl:
        dl = pd.read_csv(args.dl, parse_dates=["ds"])
        if "dl_forecast" not in dl.columns:
            raise ValueError("dl 预测需含 dl_forecast 列（DL 项目 dl_test_predictions.csv）")
        dl = dl[["unique_id", "ds", "dl_forecast"]].rename(columns={"dl_forecast": "dl"})
        merged = merged.merge(dl, on=["unique_id", "ds"], how="inner")
        models.append("dl")

    if merged.empty:
        raise ValueError("预测与测试集无共同 (unique_id, ds)，请检查日期范围与 SKU 是否对齐")
    print(f"对齐行数: {len(merged)}，SKU: {merged['unique_id'].nunique()}，"
          f"日期 {merged['ds'].min().date()} ~ {merged['ds'].max().date()}")
    print(f"参与对比模型: {models}")

    # per-SKU SMAPE —— 完全复用 utilsforecast，与统计/ML 同口径
    uf_in = merged[["unique_id", "ds", "y"] + models]
    uf_evals = uf_evaluate(uf_in, metrics=[uf_smape], models=models)
    per_sku = uf_evals[["unique_id"] + models]
    per_sku["winner"] = per_sku[models].idxmin(axis=1)
    for m in models:
        per_sku[f"diff_{m}_minus_stat"] = per_sku[m] - per_sku["stat"]

    # 各模型全局 SMAPE 与 per-SKU 平均
    means = {m: float(per_sku[m].mean()) for m in models}
    globs = {m: _smape_global(merged["y"], merged[m]) for m in models}
    dist = per_sku["winner"].value_counts().to_dict()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sku.to_csv(Path(args.out).with_name("per_sku_smape.csv"), index=False)

    print("\n" + "=" * 64)
    print("统计 vs 机器学习 vs 深度学习 · 2021 测试集最终对比")
    print("=" * 64)
    for m in models:
        print(f"  {m:<8s} per-SKU平均={means[m]:.4f}   全局SMAPE={globs[m]:.4f}（0~1 尺度）")
    print(f"per-SKU 最优分布：{'  '.join(f'{m} 赢 {dist.get(m,0)}' for m in models)}")

    summary = pd.DataFrame({
        "metric": ["mean_per_sku_smape", "global_smape"],
        **{m: [means[m], globs[m]] for m in models},
    })
    summary.to_csv(args.out, index=False)
    print(f"\n已保存汇总到: {args.out}")
    print(f"已保存 per-SKU 明细到: {Path(args.out).with_name('per_sku_smape.csv')}")

    # 决策建议（仅提示，非自动选择）
    print("\n" + "-" * 64)
    print("【使用建议】（依据训练集 CV 的模型分布，而非本测试集结果）")
    total = len(per_sku)
    if total:
        best_model = max(models, key=lambda m: dist.get(m, 0) / total)
        share = dist.get(best_model, 0) / total
        if share >= 0.8:
            print(f"→ CV 上 {best_model} 在绝大多数 SKU 占优（{share:.0%}），建议全局采用 {best_model}。")
        else:
            print(f"→ 各模型均有适用 SKU（最优分布分散），建议 per-SKU 混合（以训练集 CV 选出的最优为准）。")
    print("→ 本测试集结果仅作最终确认与展示，请勿据此回改模型（否则测试集将失效）。")


if __name__ == "__main__":
    main()
