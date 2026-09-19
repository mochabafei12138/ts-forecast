# -*- coding: utf-8 -*-
"""
blend_selection.py — 跨三范式（stat / ml / dl）per-SKU 选优 + 上线预测

只读现有三方已算好的结果，不修改 main.py / ml_project / dl_project 的任何流程。
流程：
  1) 读取三方 per-SKU 的 CV SMAPE：
       stat:  runs/<id>/model_comparison.csv  的 best_smape（统计家族内最优）
       ml:    ml_project/output/ml_cv_report.csv  的 ml_smape
       dl:    dl_project/output/dl_cv_report.csv  的 dl_smape
  2) 每个 SKU 在三方里取 SMAPE 最小者 → 跨范式 best_model
  3) 按各自范式全量重训，生成 2022 预测（默认输出到 runs/blend_<ts>/final_forecast.csv）

用法：
    python blend_selection.py                    # 全量
    python blend_selection.py --limit-skus 3     # 冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.config import load_config
from src.models.registry import build_models

DEFAULT_STAT = PROJECT_DIR / 'runs' / '20260918-113555' / 'model_comparison.csv'
DEFAULT_ML = PROJECT_DIR / 'ml_project' / 'output' / 'ml_cv_report.csv'
DEFAULT_DL = PROJECT_DIR / 'dl_project' / 'output' / 'dl_cv_report.csv'


def run_sub(args, cwd):
    """在独立进程里跑 ML / DL 辅助预测脚本，避免模块名冲突。"""
    cmd = [sys.executable] + args
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])
        raise RuntimeError(f"子进程失败: {cmd}")


def main():
    ap = argparse.ArgumentParser(description='跨三范式 per-SKU 选优 + 上线预测')
    ap.add_argument('--stat-model-comparison', default=None)
    ap.add_argument('--ml-cv-report', default=None)
    ap.add_argument('--dl-cv-report', default=None)
    ap.add_argument('--exp', default='exp_baseline')
    ap.add_argument('--limit-skus', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--prophet-cache-run', default=None,
                    help='复用哪个 run 的 Prophet 调参缓存（默认用已完成调参的 run）')
    args = ap.parse_args()

    # ---------- 数据 ----------
    train_df = pd.read_csv(PROJECT_DIR / 'data' / 'train.csv', parse_dates=['ds'])
    test_df = pd.read_csv(PROJECT_DIR / 'data' / 'test.csv', parse_dates=['ds'])
    h, level = 365, 95
    if args.limit_skus:
        skus = train_df['unique_id'].unique()[:args.limit_skus]
        train_df = train_df[train_df['unique_id'].isin(skus)]
        test_df = test_df[test_df['unique_id'].isin(skus)]

    # ---------- 读取三方 per-SKU CV 误差 ----------
    stat = pd.read_csv(args.stat_model_comparison or DEFAULT_STAT)
    ml = pd.read_csv(args.ml_cv_report or DEFAULT_ML)
    dl = pd.read_csv(args.dl_cv_report or DEFAULT_DL)

    stat = stat[['unique_id', 'best_model', 'best_smape']]
    m = (stat.merge(ml[['unique_id', 'ml_smape']], on='unique_id')
             .merge(dl[['unique_id', 'dl_smape']], on='unique_id'))
    fam_cols = ['best_smape', 'ml_smape', 'dl_smape']
    m['family'] = np.argmin(m[fam_cols].values, axis=1)
    m['family'] = m['family'].map({0: 'stat', 1: 'ml', 2: 'dl'})
    m['model_key'] = np.where(m['family'] == 'stat', m['best_model'], m['family'])
    lookup = {'stat': 'best_smape', 'ml': 'ml_smape', 'dl': 'dl_smape'}
    m['chosen_smape'] = m.apply(lambda r: r[lookup[r['family']]], axis=1)
    m['cv_smape_stat'] = m['best_smape']
    m['cv_smape_ml'] = m['ml_smape']
    m['cv_smape_dl'] = m['dl_smape']

    if args.limit_skus:
        m = m[m['unique_id'].isin(set(train_df['unique_id'].unique()))]

    print("==== 跨范式 per-SKU 选优结果（按范式） ====")
    print(m['family'].value_counts().to_string())
    print("\n统计家族内被选中的具体模型分布:")
    print(m[m['family'] == 'stat']['best_model'].value_counts().to_string())

    # ---------- 全量重训预测 2022 ----------
    out_dir = Path(args.out) if args.out else PROJECT_DIR / 'runs' / (
        'blend_' + time.strftime('%Y%m%d-%H%M%S'))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写 SKU 清单供 ML / DL 辅助脚本使用
    for fam, fname in [('ml', 'ml_skus.json'), ('dl', 'dl_skus.json')]:
        sku_list = m[m['family'] == fam]['unique_id'].tolist()
        (out_dir / fname).write_text(json.dumps(sku_list), encoding='utf-8')

    results = []

    # 1) stat 选中的 SKU
    stat_grp = m[m['family'] == 'stat']
    if not stat_grp.empty:
        models = build_models(load_config(args.exp))
        # 复用既有 Prophet 调参缓存（避免重新调参；缓存齐全则不会重算）
        prophet_cache = (Path(args.prophet_cache_run) if args.prophet_cache_run
                         else PROJECT_DIR / 'runs' / '20260918-113555')
        for mm in models:
            if getattr(mm, 'tune_enabled', False) and hasattr(mm, 'set_runs_dir'):
                mm.set_runs_dir(prophet_cache)
                try:
                    mm.tune(train_df, {}, h=h)
                    print(f"  [Prophet {mm.name}] 已加载调参缓存")
                except Exception as e:
                    print(f"  [警告] {mm.name} 加载调参缓存失败: {e}")
        stat_by = {mm.name: mm for mm in models}
        for model_key, grp in stat_grp.groupby('model_key'):
            if model_key not in stat_by:
                print(f"  [跳过] 统计模型 {model_key} 未在配置中")
                continue
            pred = stat_by[model_key].forecast(full_df(train_df, test_df), h=h, level=level)
            pred = pred[pred['unique_id'].isin(grp['unique_id'])]
            pred = pred.rename(columns={
                'yhat': 'best_forecast',
                'yhat-lo': 'best_forecast-lo-95',
                'yhat-hi': 'best_forecast-hi-95'})
            pred['best_model'] = model_key
            results.append(pred)
        print(f"  stat 选中 {len(stat_grp)} 个 SKU 已预测")

    # 2) ml 选中的 SKU（子进程）
    if (out_dir / 'ml_skus.json').exists():
        ml_skus = json.loads((out_dir / 'ml_skus.json').read_text(encoding='utf-8'))
        if ml_skus:
            run_sub(['blend_predict.py', '--skus', str(out_dir / 'ml_skus.json'),
                     '--out', str(out_dir / 'ml_final.csv'), '--data', str(PROJECT_DIR / 'data')],
                    cwd=PROJECT_DIR / 'ml_project')
            results.append(pd.read_csv(out_dir / 'ml_final.csv', parse_dates=['ds']))

    # 3) dl 选中的 SKU（子进程）
    if (out_dir / 'dl_skus.json').exists():
        dl_skus = json.loads((out_dir / 'dl_skus.json').read_text(encoding='utf-8'))
        if dl_skus:
            run_sub(['blend_predict.py', '--skus', str(out_dir / 'dl_skus.json'),
                     '--out', str(out_dir / 'dl_final.csv'), '--data', str(PROJECT_DIR / 'data')],
                    cwd=PROJECT_DIR / 'dl_project')
            results.append(pd.read_csv(out_dir / 'dl_final.csv', parse_dates=['ds']))

    if not results:
        print("没有可输出的预测结果。")
        return

    final = pd.concat(results, ignore_index=True)
    final = final.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    final.to_csv(out_dir / 'final_forecast.csv', index=False)
    m.to_csv(out_dir / 'blend_selection.csv', index=False)
    m['family'].value_counts().to_csv(out_dir / 'blend_model_distribution.csv')

    print("\n已输出到:", out_dir)
    print("跨范式 best_model 分布（按 SKU）：")
    print(final.groupby('unique_id')['best_model'].last().value_counts().to_string())
    print("\n跨范式上线 SMAPE 均值为：")
    print(f"  stat={m['cv_smape_stat'].mean():.4f}  ml={m['cv_smape_ml'].mean():.4f}  "
          f"dl={m['cv_smape_dl'].mean():.4f}  混合平均={m['chosen_smape'].mean():.4f}")


def full_df(train_df, test_df):
    return (pd.concat([train_df, test_df], ignore_index=True)
            .sort_values(['unique_id', 'ds']).reset_index(drop=True))


if __name__ == '__main__':
    main()
