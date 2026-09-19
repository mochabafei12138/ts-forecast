# -*- coding: utf-8 -*-
"""
blend_predict.py（ML 项目内）— 为跨范式选优生成 ML 选中 SKU 的 2022 全量预测
由 blend_selection.py 以子进程方式调用，不改动现有 main.py。
用法：
    python blend_predict.py --skus <json: SKU列表> --out <csv> --data <data目录>
输出统一列：unique_id, ds, best_forecast, best_forecast-lo-95, best_forecast-hi-95, best_model('ml')
"""
import argparse, json
from pathlib import Path

import pandas as pd

import model as M
import features as F
from config import TRAIN_FILE, TEST_FILE, H, STATIC_COLS, LEVEL, ML_BEST_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=None)
    args = ap.parse_args()

    sku_set = set(json.loads(Path(args.skus).read_text(encoding='utf-8')))

    # 全量数据（train+test 合并重训）
    if args.data:
        train_df = pd.read_csv(Path(args.data) / 'train.csv', parse_dates=['ds'])
        test_df = pd.read_csv(Path(args.data) / 'test.csv', parse_dates=['ds'])
    else:
        train_df = pd.read_csv(TRAIN_FILE, parse_dates=['ds'])
        test_df = pd.read_csv(TEST_FILE, parse_dates=['ds'])
    full_df = (pd.concat([train_df, test_df], ignore_index=True)
               .sort_values(['unique_id', 'ds']).reset_index(drop=True))

    # 最优超参
    bp = pd.read_csv(ML_BEST_PARAMS, index_col=0)
    best = {k: float(v) for k, v in bp.iloc[:, 0].items()}

    # 全量重训 + 预测未来 H 天
    # 启用共形预测区间（PredictionIntervals），否则 predict(level=...) 不会生成 lo/hi 列
    from mlforecast.utils import PredictionIntervals
    full_hol, _enc = M.prepare_data(full_df)
    fcst = M.make_mlforecast(best)
    fcst.fit(full_hol, static_features=STATIC_COLS,
             prediction_intervals=PredictionIntervals(n_windows=2, h=H))

    future = F.build_future_holiday_df(full_df, h=H)
    static_lookup = full_hol[['unique_id'] + STATIC_COLS].drop_duplicates()
    future = future.merge(static_lookup, on='unique_id')
    X_dynamic = future.drop(columns=STATIC_COLS, errors='ignore')
    preds = fcst.predict(H, X_df=X_dynamic, level=[LEVEL])

    preds = preds.rename(columns={
        'lgb': 'best_forecast',
        f'lgb-lo-{LEVEL}': 'best_forecast-lo-95',
        f'lgb-hi-{LEVEL}': 'best_forecast-hi-95'})
    preds['ds'] = pd.to_datetime(preds['ds'])
    keep = ['unique_id', 'ds', 'best_forecast']
    if 'best_forecast-lo-95' in preds.columns:
        keep += ['best_forecast-lo-95', 'best_forecast-hi-95']
    preds = preds[keep]
    preds = preds[preds['unique_id'].isin(sku_set)]
    preds['best_model'] = 'ml'
    preds = preds.sort_values(['unique_id', 'ds'])
    preds.to_csv(args.out, index=False)
    print(f"[ml] 已预测 {preds['unique_id'].nunique()} 个 SKU -> {args.out}")


if __name__ == '__main__':
    main()
