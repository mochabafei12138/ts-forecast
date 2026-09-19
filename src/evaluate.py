# -*- coding: utf-8 -*-
"""
evaluate.py — 统一评估与选优
1) CV 结果合并、per-SKU 选最优模型
2) 在独立测试集上评估
3) 合并 train+test 重训，输出上线预测
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape as uf_smape


def _available_models(cv_merged: pd.DataFrame) -> list[str]:
    return [c for c in cv_merged.columns
            if c not in ('unique_id', 'ds', 'cutoff', 'y')]


def evaluate_cv(all_cv: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    合并各模型的 CV 结果，统一 SMAPE 评估，选 per-SKU 最优。
    返回 evaluation_df，列含各模型名 + best_model + best_smape。
    """
    model_names = [n for n, df in all_cv.items()
                   if df is not None and not df.empty]
    merged: pd.DataFrame | None = None
    for name in model_names:
        cv = all_cv[name].rename(columns={'yhat': name})
        if merged is None:
            # 第一次取基准键 + 真实值 y + 第一个模型列，避免后续 y 冲突
            merged = cv[['unique_id', 'ds', 'cutoff', 'y', name]].copy()
            continue
        merged = merged.merge(
            cv[['unique_id', 'ds', 'cutoff', name]],
            on=['unique_id', 'ds', 'cutoff'], how='left')

    models = _available_models(merged)
    if len(models) < 2:
        raise ValueError(f"可用于评估的模型列不足: {models}")

    # drop cutoff，避免 utilsforecast 把它当作分组 id 导致每个 SKU 多行
    eval_input = merged.drop(columns=['cutoff'], errors='ignore')
    evals = evaluate(eval_input, metrics=[uf_smape], models=models)
    evals = evals.drop(columns=['metric'])
    evals['best_model'] = evals[models].idxmin(axis=1)
    evals['best_smape'] = evals.apply(
        lambda r: r[r['best_model']], axis=1)
    return evals


def get_best_model_dist(evals: pd.DataFrame) -> dict:
    return evals['best_model'].value_counts().to_dict()


# ---------------- 测试集评估 ----------------
def evaluate_on_test_set(train_df, test_df, evals,
                         models: dict[str, object], h: int, level: int,
                         pred_save_path: str | None = None) -> pd.DataFrame:
    """用 per-SKU 最优模型在独立测试集上评估。
    若给定 pred_save_path，则把测试集逐日预测（unique_id, ds, best_forecast）一并保存。"""
    rows = []
    test_preds: list[pd.DataFrame] = []
    # 按最优模型分组，批量预测
    for best_model, grp in evals.groupby('best_model'):
        if best_model not in models:
            continue
        skus = grp['unique_id'].tolist()
        pred = models[best_model].forecast(train_df, h=h, level=level)
        pred = pred[pred['unique_id'].isin(skus)]
        pred = pred.rename(columns={'yhat': 'best_forecast'})
        test_preds.append(pred[['unique_id', 'ds', 'best_forecast']].copy())
        for uid in tqdm(skus, desc=f"评估 {best_model}"):
            test_true = test_df[test_df['unique_id'] == uid][['ds', 'y']]
            test_true = test_true.rename(columns={'y': 'y_true'})
            p = pred[pred['unique_id'] == uid][['ds', 'best_forecast']]
            merged = test_true.merge(p, on='ds', how='inner')
            if merged.empty:
                continue
            eval_df = pd.DataFrame({
                'unique_id': uid, 'ds': merged['ds'],
                'y': merged['y_true'].values,
                'best_model_pred': merged['best_forecast'].values,
            })
            try:
                ev = evaluate(eval_df, metrics=[uf_smape],
                              models=['best_model_pred'])
                smape_val = float(ev['best_model_pred'].iloc[0])
            except Exception:
                smape_val = np.nan
            rows.append({
                'unique_id': uid, 'best_model': best_model,
                'test_smape': smape_val, 'test_n': len(merged),
            })
    if pred_save_path and test_preds:
        all_preds = pd.concat(test_preds, ignore_index=True)
        all_preds.to_csv(pred_save_path, index=False)
        print(f"已保存测试集逐日预测到: {pred_save_path}（{len(all_preds)} 行）")
    return pd.DataFrame(rows)


# ---------------- 上线预测（合并重训） ----------------
def retrain_with_full_data(train_df, test_df, evals,
                           models: dict[str, object], h: int, level: int
                           ) -> pd.DataFrame:
    """合并 train+test 全量数据，用每 SKU 最优模型重训，输出未来 h 天。"""
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    results = []
    for best_model, grp in evals.groupby('best_model'):
        if best_model not in models:
            continue
        skus = grp['unique_id'].tolist()
        try:
            pred = models[best_model].forecast(full_df, h=h, level=level)
        except Exception:
            continue
        pred = pred[pred['unique_id'].isin(skus)]
        pred = pred.rename(columns={
            'yhat': 'best_forecast',
            'yhat-lo': 'best_forecast-lo-95',
            'yhat-hi': 'best_forecast-hi-95',
        })
        pred['best_model'] = best_model
        results.append(pred)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)
