# -*- coding: utf-8 -*-
"""
main.py — 实验运行入口（可持续迭代框架）
用法:
    python main.py                          # 运行第一个实验配置
    python main.py --exp exp_baseline        # 指定实验
    python main.py --resume 20260911-120000  # 断点续跑
    python main.py --limit-skus 5            # 仅跑前 N 个 SKU（冒烟/调试）
    python main.py --note "改了模型X"        # 备注

流程: 配置 -> 模型构建 -> 数据 -> 各模型 CV(缓存) -> 统一评估选优
      -> 测试集评估 -> 合并重训上线预测 -> 写入 SQLite 与 runs/<id>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import load_config, PROJECT_DIR
from src.data.loader import load_data, get_all_sku_ids
from src.models.registry import build_models
from src.cache import CacheStore
from src.experiment import ExperimentTracker
from src import evaluate as ev


def parse_args():
    p = argparse.ArgumentParser(description='时间序列预测实验运行器')
    p.add_argument('--exp', type=str, default=None,
                   help='实验配置名（configs/experiments 下的文件名，不含 .yaml）')
    p.add_argument('--resume', type=str, default=None,
                   help='断点续跑：传入 run_id，跳过已完成阶段')
    p.add_argument('--limit-skus', type=int, default=None,
                   help='仅处理前 N 个 SKU（冒烟测试用）')
    p.add_argument('--note', type=str, default='', help='本次运行备注')
    return p.parse_args()


def _to_serializable_cutoffs(cm: dict) -> dict:
    return {u: [str(c) for c in cutoffs] for u, cutoffs in cm.items()}


def _from_serializable_cutoffs(raw: dict) -> dict:
    return {u: pd.to_datetime([pd.Timestamp(c) for c in cutoffs])
            for u, cutoffs in raw.items()}


def main():
    t0 = time.time()
    args = parse_args()
    cfg = load_config(args.exp)

    runs_dir = Path(cfg.get('paths', 'runs_dir', default=str(PROJECT_DIR / 'runs')))
    db_file = cfg.get('paths', 'db_file', default=str(PROJECT_DIR / 'experiments.db'))
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 确定 run 目录（resume 复用已有） ----------
    run_id = args.resume or time.strftime('%Y%m%d-%H%M%S')
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    (run_dir / 'args.json').write_text(
        json.dumps(vars(args), indent=2), encoding='utf-8')
    print(f"Run ID: {run_id}  目录: {run_dir}")

    tracker = ExperimentTracker(db_file, project_dir=PROJECT_DIR)
    if args.resume is None:
        tracker.start_run(cfg.get('experiment', 'name', default='unnamed'),
                          cfg.to_dict(), note=args.note)

    # ---------- 构建模型 ----------
    models = build_models(cfg)
    by_name = {m.name: m for m in models}
    for m in models:
        if hasattr(m, 'set_runs_dir'):
            m.set_runs_dir(run_dir)
    print(f"参与实验的模型: {[(m.name, type(m).__name__) for m in models]}")

    cache = CacheStore(run_dir)

    # ---------- 数据 ----------
    train_df, test_df, info = load_data(
        cfg.get('data', 'train_file'), cfg.get('data', 'test_file'))
    if args.limit_skus:
        skus = get_all_sku_ids(train_df)[:args.limit_skus]
        train_df = train_df[train_df['unique_id'].isin(skus)]
        test_df = test_df[test_df['unique_id'].isin(skus)]
        print(f"[冒烟] 仅使用前 {args.limit_skus} 个 SKU")

    h = cfg.get('forecast', 'h', default=365)
    level = cfg.get('forecast', 'level', default=95)
    n_windows = cfg.get('cv', 'n_windows', default=2)
    step_size = cfg.get('cv', 'step_size', default=365)

    # ---------- Step 1: 各模型 CV（带断点缓存） ----------
    print("\n" + "=" * 60 + "\nStep 1: 交叉验证\n" + "=" * 60)
    all_cv: dict[str, pd.DataFrame] = {}
    cutoffs_map = None

    # Pass 1: 自动生成 cutoffs 的统计模型
    for m in models:
        if m.needs_cutoffs:
            continue
        key = f'cv_{m.name}'
        if cache.exists('cv', key):
            all_cv[m.name] = cache.load('cv', key)
            print(f"  [缓存] 跳过 {m.name} 的 CV")
            if cutoffs_map is None:
                raw = cache.load_json('cv', 'cutoffs_map')
                if raw:
                    cutoffs_map = _from_serializable_cutoffs(raw)
            continue
        print(f"  运行 {m.name} CV ...")
        cv_df, cm = m.cv(train_df, h=h, level=level,
                         n_windows=n_windows, step_size=step_size,
                         cutoffs_map=cutoffs_map)
        all_cv[m.name] = cv_df
        cache.save('cv', key, cv_df)
        if cm is not None:
            cutoffs_map = cm
            cache.save_json('cv', 'cutoffs_map', _to_serializable_cutoffs(cm))

    # Pass 2: 需要外部 cutoffs 的模型（Prophet）
    for m in models:
        if not m.needs_cutoffs:
            continue
        key = f'cv_{m.name}'
        if cache.exists('cv', key):
            all_cv[m.name] = cache.load('cv', key)
            print(f"  [缓存] 跳过 {m.name} 的 CV")
            continue
        if cutoffs_map is None:
            raw = cache.load_json('cv', 'cutoffs_map')
            cutoffs_map = _from_serializable_cutoffs(raw) if raw else None
        if cutoffs_map is None:
            raise RuntimeError(
                "需要先注册一个统计模型以提供 cutoffs（Prophet 依赖它）")
        print(f"  运行 {m.name} CV ...")
        cv_df, _ = m.cv(train_df, h=h, level=level,
                        n_windows=n_windows, step_size=step_size,
                        cutoffs_map=cutoffs_map)
        all_cv[m.name] = cv_df
        cache.save('cv', key, cv_df)

    # ---------- Step 2: 统一评估 + 选优 ----------
    print("\n" + "=" * 60 + "\nStep 2: 统一评估（CV SMAPE）\n" + "=" * 60)
    evals = ev.evaluate_cv(all_cv)
    dist = ev.get_best_model_dist(evals)
    print("各 SKU 最优模型分布:")
    for m, c in dist.items():
        print(f"  {m:<25s}: {c}")
    print(f"per-SKU 最优平均 SMAPE = {evals['best_smape'].mean():.4f}%")

    # 模型级摘要（CV 阶段）：仅保留数值型的模型列
    _exclude = {'unique_id', 'best_model', 'best_smape', 'cutoff', 'ds', 'metric'}
    model_cols = [c for c in evals.columns
                  if c not in _exclude
                  and pd.api.types.is_numeric_dtype(evals[c])]
    model_summary = evals[model_cols].mean().reset_index()
    model_summary.columns = ['model', 'mean_smape']
    model_summary['n_sku'] = evals['best_model'].value_counts().reindex(
        model_summary['model']).fillna(0).astype(int).values
    print("\n模型级对比（全部 SKU 平均 SMAPE）:")
    print(model_summary.sort_values('mean_smape'))

    # ---------- Step 3: 各模型全量预测（缓存） ----------
    print("\n" + "=" * 60 + "\nStep 3: 各模型全量预测\n" + "=" * 60)
    forecasts: dict[str, pd.DataFrame] = {}
    for m in models:
        key = f'forecast_{m.name}'
        if cache.exists('fc', key):
            forecasts[m.name] = cache.load('fc', key)
            print(f"  [缓存] 跳过 {m.name} 预测")
            continue
        fc = m.forecast(train_df, h=h, level=level)
        forecasts[m.name] = fc
        cache.save('fc', key, fc)

    # ---------- Step 4: 测试集评估 ----------
    print("\n" + "=" * 60 + "\nStep 4: 测试集评估\n" + "=" * 60)
    test_eval = ev.evaluate_on_test_set(
        train_df, test_df, evals, by_name, h, level,
        pred_save_path=str(run_dir / 'test_set_predictions.csv'))
    if not test_eval.empty:
        print(f"测试集平均 SMAPE = {test_eval['test_smape'].mean():.4f}%")

    # ---------- Step 5: 合并重训上线预测 ----------
    print("\n" + "=" * 60 + "\nStep 5: 合并全量重训，上线预测\n" + "=" * 60)
    final_forecast = ev.retrain_with_full_data(
        train_df, test_df, evals, by_name, h, level)

    # ---------- 汇总输出 ----------
    metrics = evals.rename(columns={'best_smape': 'cv_smape'})[
        ['unique_id', 'best_model', 'cv_smape']]
    if not test_eval.empty:
        metrics = metrics.merge(
            test_eval[['unique_id', 'test_smape', 'test_n']],
            on='unique_id', how='left')

    tracker.save_metrics(run_id, metrics)
    tracker.save_model_summary(run_id, model_summary, stage='cv')
    tracker.save_best_model_dist(run_id, dist)

    # 保存结果文件到 runs/<id>
    evals.to_csv(run_dir / 'model_comparison.csv', index=False)
    if not test_eval.empty:
        test_eval.to_csv(run_dir / 'test_set_evaluation.csv', index=False)
    if not final_forecast.empty:
        final_forecast.to_csv(run_dir / 'final_forecast.csv', index=False)
    metrics.to_csv(run_dir / 'sku_smape_detail.csv', index=False)
    pd.Series(dist, name='count').to_csv(run_dir / 'best_model_distribution.csv')

    total_time = time.time() - t0
    tracker.finish_run(run_id, 'completed', total_time)
    tracker.close()

    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)
    print(f"Run ID: {run_id}  总耗时: {total_time:.1f}s")
    print(f"结果与缓存目录: {run_dir}")
    print(f"实验追踪库: {db_file}")


if __name__ == '__main__':
    main()
