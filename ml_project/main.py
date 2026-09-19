# -*- coding: utf-8 -*-
"""
main.py — ML 项目主流程

1) 训练集(2017-2020)内做时序 CV 调参（不触碰测试集）
2) 用最优超参重跑 CV，输出每 SKU 的 ML SMAPE（ml_cv_report.csv）
3) 用训练集训练最优 ML，生成 2021 测试期逐日预测（ml_test_predictions.csv，供最终对比）

用法：
    python main.py                 # 全量
    python main.py --limit-skus 3  # 仅前 3 个 SKU（冒烟测试）
"""
import argparse

import pandas as pd
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate

from config import (TRAIN_FILE, TEST_FILE, H, STATIC_COLS, LEVEL,
                    ML_CV_REPORT, ML_BEST_PARAMS, ML_TEST_PRED)
import features as F
import model as M


def load():
    train = pd.read_csv(TRAIN_FILE, parse_dates=["ds"])
    test = pd.read_csv(TEST_FILE, parse_dates=["ds"])
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-skus", type=int, default=None)
    args = ap.parse_args()

    train_df, test_df = load()
    if args.limit_skus:
        skus = train_df["unique_id"].unique()[:args.limit_skus]
        train_df = train_df[train_df["unique_id"].isin(skus)]
        test_df = test_df[test_df["unique_id"].isin(skus)]
        print(f"[冒烟] 仅使用前 {args.limit_skus} 个 SKU")

    print(f"训练集 {train_df['ds'].min().date()} ~ {train_df['ds'].max().date()}  SKU={train_df['unique_id'].nunique()}")
    print(f"测试集 {test_df['ds'].min().date()} ~ {test_df['ds'].max().date()}  （仅最终对比使用，本次不触碰用于调参）")

    # 1. 特征 + 编码
    train_hol, encoders = M.prepare_data(train_df)

    # 2. 训练集内 CV 调参
    print("\n[1/3] 网格搜索最优超参（时序 CV，n_windows=2，h=365）...")
    results, best_params = M.grid_search_cv(train_hol)
    print("最优超参:", {k: v for k, v in best_params.items()})
    print(f"CV 最优 mean SMAPE = {results['mean_smape'].iloc[0]:.4f}%")
    pd.Series(best_params).to_csv(ML_BEST_PARAMS)

    # 3. 最优超参重跑 CV，输出每 SKU 的 ML SMAPE
    print("\n[2/3] 用最优超参重跑 CV，输出每 SKU 的 ML SMAPE...")
    fcst = M.make_mlforecast(best_params)
    cv = fcst.cross_validation(train_hol, n_windows=M.N_WINDOWS, h=H,
                               static_features=STATIC_COLS, step_size=M.STEP_SIZE)
    ml_smape = evaluate(cv, metrics=[smape], models=["lgb"])[["unique_id", "lgb"]]
    ml_smape = ml_smape.rename(columns={"lgb": "ml_smape"})
    print(f"ML per-SKU CV 平均 SMAPE = {ml_smape['ml_smape'].mean():.4f}%")
    ml_smape.to_csv(ML_CV_REPORT, index=False)
    print(f"已保存到: {ML_CV_REPORT}")

    # 4. 用训练集训练最优 ML，生成 2021 测试期逐日预测（最终对比脚本输入）
    print("\n[3/3] 用训练集训练最优 ML，生成 2021 测试期逐日预测...")
    fcst_fit = M.make_mlforecast(best_params)
    fcst_fit.fit(train_hol, static_features=STATIC_COLS)

    future = F.build_future_holiday_df(train_df, h=H)
    static_lookup = train_hol[["unique_id"] + STATIC_COLS].drop_duplicates()
    future = future.merge(static_lookup, on="unique_id")
    X_dynamic = future.drop(columns=STATIC_COLS, errors="ignore")
    preds = fcst_fit.predict(H, X_df=X_dynamic, level=[LEVEL])

    # 统一列名（兼容 lgb-lo-95 / lgb-hi-95 命名）
    preds = preds.rename(columns={
        "lgb": "ml_forecast",
        f"lgb-lo-{LEVEL}": "ml_lo",
        f"lgb-hi-{LEVEL}": "ml_hi",
    })
    keep = ["unique_id", "ds", "ml_forecast"]
    if "ml_lo" in preds.columns:
        keep += ["ml_lo"]
    if "ml_hi" in preds.columns:
        keep += ["ml_hi"]
    preds["ds"] = pd.to_datetime(preds["ds"])
    preds = preds[keep].sort_values(["unique_id", "ds"])
    preds.to_csv(ML_TEST_PRED, index=False)
    print(f"已保存 2021 测试期逐日预测到: {ML_TEST_PRED}（{len(preds)} 行）")
    print("\nML 项目完成。最终 stat vs ML 对比请运行 final_compare/main.py")


if __name__ == "__main__":
    main()
