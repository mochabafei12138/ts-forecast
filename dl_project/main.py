# -*- coding: utf-8 -*-
"""
main.py — 深度学习（N-BEATS / N-HiTS / TiDE）项目主流程

1) 训练集(2017-2020)内时序 CV：
   - NBEATS 用固定超参
   - NHITS / TiDE 各做小规模参数搜索（选全局最优超参）
2) 用各模型最优超参跑 CV → 每 SKU 各模型的 SMAPE
3) 每 SKU 在 DL 家族（NBEATS / NHITS / TiDE）内选最优 → dl_smape + best_dl_model
4) 用训练集按 per-SKU 最优 DL 模型生成 2021 测试期逐日预测

用法：
    python main.py                 # 全量（含小规模调参）
    python main.py --limit-skus 3  # 冒烟
    python main.py --skip-search   # 跳过调参，用默认超参（快速验证流程）
"""
import argparse
import json

import pandas as pd
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate

from config import (TRAIN_FILE, TEST_FILE, OUTPUT_DIR, DL_CV_REPORT,
                    DL_TEST_PRED, DL_GRID)
import model as M


def load():
    train = pd.read_csv(TRAIN_FILE, parse_dates=["ds"])
    test = pd.read_csv(TEST_FILE, parse_dates=["ds"])
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-skus", type=int, default=None)
    ap.add_argument("--skip-search", action="store_true",
                    help="跳过调参，直接用默认超参（快速验证流程用）")
    args = ap.parse_args()

    train_df, test_df = load()
    if args.limit_skus:
        skus = train_df["unique_id"].unique()[:args.limit_skus]
        train_df = train_df[train_df["unique_id"].isin(skus)]
        test_df = test_df[test_df["unique_id"].isin(skus)]
        print(f"[冒烟] 仅使用前 {args.limit_skus} 个 SKU")

    print(f"训练集 {train_df['ds'].min().date()} ~ {train_df['ds'].max().date()}  SKU={train_df['unique_id'].nunique()}")
    print(f"测试集 {test_df['ds'].min().date()} ~ {test_df['ds'].max().date()}  （仅最终对比使用）")

    # ---------- 1. 各模型 CV + 调参 ----------
    print("\n[1/3] 各 DL 模型时序 CV / 小规模调参 ...")
    best_params = {"NBEATS": {}, "NHITS": {}, "TiDE": {}}  # 各模型最优超参
    per_model_smape = {}                  # 各模型每 SKU 的 smape

    for model_name in ["NBEATS", "NHITS", "TiDE"]:
        if model_name != "NBEATS" and not args.skip_search:
            grid = DL_GRID[model_name]
            print(f"  [{model_name}] 网格搜索（{len(grid)} 组）...")
            res, best = M.grid_search_cv(train_df, model_name, grid)
            print(f"    最优超参: {best}   CV mean SMAPE={res['mean_smape'].iloc[0]:.4f}")
            best_params[model_name] = best
        # 用最优/默认超参跑 CV，得到每 SKU 的 smape
        model = M.DL_MODELS[model_name](**best_params[model_name])
        cv = M.run_cv(train_df, model=model)
        smp = evaluate(cv, metrics=[smape], models=[model_name])[
            ["unique_id", model_name]].rename(columns={model_name: f"{model_name}_smape"})
        per_model_smape[model_name] = smp
        print(f"  [{model_name}] 每 SKU CV SMAPE 平均 = {smp[f'{model_name}_smape'].mean():.4f}")

    # ---------- 2. DL 家族 per-SKU 选优 ----------
    print("\n[2/3] DL 家族 per-SKU 选优（NBEATS / NHITS / TiDE 取最优）...")
    smape_cols = ["NBEATS_smape", "NHITS_smape", "TiDE_smape"]
    merged = per_model_smape["NBEATS"]
    for mn in ["NHITS", "TiDE"]:
        merged = merged.merge(per_model_smape[mn], on="unique_id", how="left")
    merged["dl_smape"] = merged[smape_cols].min(axis=1)
    merged["best_dl_model"] = merged[smape_cols].idxmin(axis=1).str.replace("_smape", "")
    dl_smape = merged[["unique_id", "dl_smape", "best_dl_model"] + smape_cols]
    dl_smape.to_csv(DL_CV_REPORT, index=False)
    # 保存各模型最优超参，供 blend_predict 重建模型预测
    (OUTPUT_DIR / "dl_best_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DL per-SKU 最优 SMAPE 平均 = {dl_smape['dl_smape'].mean():.4f}")
    print("DL 家族模型分布:\n", dl_smape["best_dl_model"].value_counts().to_string())
    print(f"已保存到: {DL_CV_REPORT}")

    # ---------- 3. 测试期逐日预测（per-SKU 最优 DL 模型） ----------
    print("\n[3/3] 用 per-SKU 最优 DL 模型生成 2021 测试期逐日预测...")
    preds_list = []
    for mn in ["NBEATS", "NHITS", "TiDE"]:
        skus = dl_smape[dl_smape["best_dl_model"] == mn]["unique_id"].tolist()
        if not skus:
            continue
        model = M.DL_MODELS[mn](**best_params[mn])
        pred = M.forecast(train_df, model=model)
        pred = pred[pred["unique_id"].isin(skus)]
        pred = pred.rename(columns={mn: "dl_forecast"})
        preds_list.append(pred)
    preds = pd.concat(preds_list, ignore_index=True)
    preds = preds[["unique_id", "ds", "dl_forecast"]].sort_values(["unique_id", "ds"])
    preds.to_csv(DL_TEST_PRED, index=False)
    print(f"已保存 2021 测试期逐日预测到: {DL_TEST_PRED}（{len(preds)} 行）")
    print("\nDL 项目完成。最终对比请运行 final_compare/main.py（--dl 指定本文件）")


if __name__ == "__main__":
    main()
