# -*- coding: utf-8 -*-
"""
prophet_model.py — Prophet 模型 wrapper
支持 有/无节假日、超参数网格调参（结果缓存到 runs 目录，可断点续跑）。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import ForecastModel
from .registry import register_factory

_DEFAULT_PARAM_GRID = {
    'changepoint_prior_scale': [0.01, 0.1],
    'seasonality_prior_scale': [5.0, 10.0],
    'yearly_seasonality': [True, False],
    'weekly_seasonality': [True, False],
    'daily_seasonality': [False],
    'seasonality_mode': ['additive', 'multiplicative'],
}


class ProphetModel(ForecastModel):
    needs_cutoffs = True

    def __init__(self, name: str, use_holidays: bool = True,
                 tune: bool = False, param_grid: dict | None = None,
                 interval_width: float = 0.95):
        self.name = name
        self.use_holidays = use_holidays
        self.tune_enabled = tune
        self.param_grid = param_grid or dict(_DEFAULT_PARAM_GRID)
        self.interval_width = interval_width
        self.best_params_map: dict | None = None
        self.runs_dir: Path | None = None

    def set_runs_dir(self, path):
        self.runs_dir = Path(path)

    # ---------------- 工具 ----------------
    def _combinations(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        return [
            dict(zip(keys, vals))
            for vals in itertools.product(*(self.param_grid[k] for k in keys))
        ]

    def _build(self, params: dict, train_df: pd.DataFrame, uid: str):
        from prophet import Prophet
        sku = train_df[train_df['unique_id'] == uid].sort_values('ds')
        country = None
        if self.use_holidays and 'country' in sku.columns and len(sku):
            country = sku['country'].iloc[0]
        model = Prophet(
            weekly_seasonality=params['weekly_seasonality'],
            yearly_seasonality=params['yearly_seasonality'],
            daily_seasonality=params['daily_seasonality'],
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            interval_width=self.interval_width,
        )
        if self.use_holidays and country is not None:
            try:
                model.add_country_holidays(country_name=country)
            except Exception:
                pass
        return model

    def _prophet_cv(self, train_df, uid, params, cutoffs, h):
        from prophet.diagnostics import cross_validation
        model = self._build(params, train_df, uid)
        sku = train_df[train_df['unique_id'] == uid].sort_values('ds')
        try:
            model.fit(sku[['ds', 'y']])
        except Exception:
            return None
        try:
            return cross_validation(model, cutoffs=cutoffs,
                                    horizon=f'{h} days', disable_tqdm=True)
        except Exception:
            valid = [c for c in cutoffs if c < sku['ds'].max()]
            if len(valid) < len(cutoffs):
                try:
                    return cross_validation(model, cutoffs=valid,
                                            horizon=f'{h} days',
                                            disable_tqdm=True)
                except Exception:
                    return None
            return None

    @staticmethod
    def _smape(cv_df) -> float:
        """从 prophet cross_validation 结果计算 SMAPE。"""
        if cv_df is None or cv_df.empty:
            return np.nan
        from utilsforecast.evaluation import evaluate
        from utilsforecast.losses import smape
        pred = cv_df[['cutoff', 'ds', 'y', 'yhat']].copy()
        pred['unique_id'] = 'sku'
        pred = pred.rename(columns={'yhat': 'model'})
        try:
            evals = evaluate(pred, metrics=[smape], models=['model'])
            val = float(evals['model'].iloc[0])
            return val if not np.isnan(val) else np.nan
        except Exception:
            return np.nan

    # ---------------- 调参（带缓存） ----------------
    def tune(self, train_df: pd.DataFrame, cutoffs_map: dict,
             h: int = 365, use_cache: bool = True) -> dict:
        if self.runs_dir is None:
            raise RuntimeError("调参前需先 set_runs_dir()")
        cache_dir = self.runs_dir / 'prophet_params'
        cache_dir.mkdir(parents=True, exist_ok=True)

        best_params_map: dict = {}
        combos = self._combinations()
        print(f"  [Prophet {self.name}] 调参搜索空间 {len(combos)} 种组合")

        for uid in train_df['unique_id'].unique():
            cache_file = cache_dir / f"{self.name}_{uid}.json"
            # 断点续跑：已有缓存则直接读取
            if use_cache and cache_file.exists():
                try:
                    best_params_map[uid] = json.loads(
                        cache_file.read_text(encoding='utf-8'))
                    continue
                except Exception:
                    pass
            cutoffs = cutoffs_map.get(uid)
            if not cutoffs:
                best_params_map[uid] = None
                continue
            best_metric = np.inf
            best_params = None
            for params in combos:
                cv_df = self._prophet_cv(train_df, uid, params, cutoffs, h)
                if cv_df is None or cv_df.empty:
                    continue
                metric = self._smape(cv_df)
                if np.isnan(metric):
                    continue
                if metric < best_metric:
                    best_metric = metric
                    best_params = params
            best_params_map[uid] = best_params
            # 写缓存
            try:
                cache_file.write_text(
                    json.dumps(best_params), encoding='utf-8')
            except Exception:
                pass
        self.best_params_map = best_params_map
        return best_params_map

    # ---------------- 预测 ----------------
    def forecast(self, train_df: pd.DataFrame, h: int, level: int = 95
                 ) -> pd.DataFrame:
        if self.tune_enabled and self.best_params_map is None:
            raise RuntimeError("Prophet 调参模型需先执行 tune() 生成预测参数")
        rows = []
        for uid, sku in train_df.groupby('unique_id'):
            params = self._resolve_params(uid)
            if params is None:
                continue
            model = self._build(params, train_df, uid)
            try:
                model.fit(sku.sort_values('ds')[['ds', 'y']])
            except Exception:
                continue
            future = model.make_future_dataframe(periods=h, freq='D')
            pred = model.predict(future)
            last = sku['ds'].max()
            future_part = pred[pred['ds'] > last].head(h)
            rows.append(pd.DataFrame({
                'unique_id': uid,
                'ds': future_part['ds'].values,
                'yhat': future_part['yhat'].values,
                'yhat-lo': future_part['yhat_lower'].values,
                'yhat-hi': future_part['yhat_upper'].values,
            }))
        if not rows:
            return pd.DataFrame(
                columns=['unique_id', 'ds', 'yhat', 'yhat-lo', 'yhat-hi'])
        return pd.concat(rows, ignore_index=True)

    # ---------------- CV ----------------
    def cv(self, train_df: pd.DataFrame, h: int, level: int = 95,
           n_windows: int = 2, step_size: int = 365,
           cutoffs_map: dict | None = None
           ) -> tuple[pd.DataFrame, dict | None]:
        if cutoffs_map is None:
            raise ValueError("Prophet 模型 cv() 需要外部 cutoffs_map")
        if self.tune_enabled and self.best_params_map is None:
            self.tune(train_df, cutoffs_map, h=h)
        rows = []
        for uid, sku in train_df.groupby('unique_id'):
            params = self._resolve_params(uid)
            cutoffs = cutoffs_map.get(uid)
            if params is None or not cutoffs:
                continue
            cv_df = self._prophet_cv(train_df, uid, params, cutoffs, h)
            if cv_df is None or cv_df.empty:
                continue
            out = pd.DataFrame({
                'unique_id': uid,
                'ds': cv_df['ds'].values,
                'cutoff': cv_df['cutoff'].values,
                'y': cv_df['y'].values,
                'yhat': cv_df['yhat'].values,
            })
            rows.append(out)
        if not rows:
            return pd.DataFrame(
                columns=['unique_id', 'ds', 'cutoff', 'y', 'yhat']), cutoffs_map
        return pd.concat(rows, ignore_index=True), cutoffs_map

    def _resolve_params(self, uid: str) -> dict | None:
        if self.tune_enabled:
            return (self.best_params_map or {}).get(uid)
        # 非调参模式：用默认参数
        return {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
        }


def _make_prophet(model_cfg: dict, global_cfg) -> ProphetModel:
    pg = global_cfg.get('prophet', default={})
    param_grid = model_cfg.get('param_grid') or pg.get('param_grid')
    interval = model_cfg.get('interval_width') or pg.get('interval_width', 0.95)
    return ProphetModel(
        name=model_cfg.get('name', 'Prophet'),
        use_holidays=model_cfg.get('use_holidays', True),
        tune=model_cfg.get('tune', False),
        param_grid=param_grid,
        interval_width=interval,
    )


register_factory('prophet')(_make_prophet)
