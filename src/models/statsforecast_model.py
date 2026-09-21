# -*- coding: utf-8 -*-
"""
statsforecast_model.py — StatsForecast 统计模型 wrapper
支持 AutoETS / AutoARIMA / AutoTheta / AutoCES 等。
"""
from __future__ import annotations

from functools import partial

import pandas as pd

from .base import ForecastModel
from .registry import register_factory

# StatsForecast 模型名 -> 支持的关键字参数
_MODEL_ARGS = {
    'AutoETS': 'season_length',
    'AutoARIMA': 'season_length',
    'AutoTheta': 'season_length',
    'AutoCES': 'season_length',
}


class StatsForecastModel(ForecastModel):
    needs_cutoffs = False

    def __init__(self, name: str, model_class: str, season_length: int = 7):
        self.name = name
        self.model_class = model_class
        self.season_length = season_length

    def _make_sf(self):
        from statsforecast.core import StatsForecast
        import statsforecast.models as sfm
        cls = getattr(sfm, self.model_class)
        # 兼容不同模型的关键字参数签名
        try:
            model = cls(season_length=self.season_length)
        except TypeError:
            model = cls()
        return StatsForecast(
            models=[model], freq='D', n_jobs=-1, verbose=False,
        )

    def cv(self, train_df: pd.DataFrame, h: int, level: int = 95,
           n_windows: int = 2, step_size: int = 365,
           cutoffs_map: dict | None = None
           ) -> tuple[pd.DataFrame, dict | None]:
        cutoffs_map_out: dict = {}
        rows: list[pd.DataFrame] = []
        for uid, grp in train_df.groupby('unique_id'):
            sf = self._make_sf()
            data = grp[['ds', 'unique_id', 'y']].sort_values('ds')
            try:
                cv = sf.cross_validation(
                    df=data, h=h, step_size=step_size, n_windows=n_windows,
                )
            except Exception:
                try:
                    cv = sf.cross_validation(
                        df=data, h=h, step_size=step_size,
                        n_windows=max(n_windows - 1, 1),
                    )
                except Exception:
                    continue
            cv = cv.reset_index()
            cutoffs_map_out[uid] = sorted(cv['cutoff'].unique())
            # statsforecast 各模型的预测列名并不总等于类名
            # （例如 AutoCES 的输出列是 'CES' 而非 'AutoCES'），
            # 故动态识别“非标识/非真实值列”作为预测列，再重命名为 yhat。
            # 注意：reset_index() 会引入一个名为 'index' 的列（即行号），
            # 必须排除，否则会把行号当成预测值（早期版本曾因此让 Auto* 模型
            # 对每个 SKU 输出完全相同的伪预测，导致 SMAPE 失真）。
            pred_col = next(
                (c for c in cv.columns
                 if c not in ('unique_id', 'ds', 'cutoff', 'y', 'index')), None)
            if pred_col is not None:
                cv = cv.rename(columns={pred_col: 'yhat'})
            cols = [c for c in ['unique_id', 'ds', 'cutoff', 'y', 'yhat']
                    if c in cv.columns]
            rows.append(cv[cols])
        if not rows:
            return pd.DataFrame(), cutoffs_map_out
        cv_df = pd.concat(rows, ignore_index=True)
        return cv_df, cutoffs_map_out

    def forecast(self, train_df: pd.DataFrame, h: int, level: int = 95
                 ) -> pd.DataFrame:
        sf = self._make_sf()
        fc = sf.forecast(
            df=train_df[['ds', 'unique_id', 'y']], h=h, level=[level],
        )
        fc = fc.reset_index()
        fc['ds'] = pd.to_datetime(fc['ds'])
        # 主预测列：非标识列中不带 -lo-/-hi- 后缀的列（列名不总等于类名，如 AutoCES→'CES'）
        main_col = next(
            (c for c in fc.columns
             if c not in ('unique_id', 'ds', 'index')
             and '-lo-' not in str(c) and '-hi-' not in str(c)), None)
        if main_col is None:
            return pd.DataFrame(
                columns=['unique_id', 'ds', 'yhat', 'yhat-lo', 'yhat-hi'])
        lo_col = f'{main_col}-lo-{level}'
        hi_col = f'{main_col}-hi-{level}'
        return pd.DataFrame({
            'unique_id': fc['unique_id'],
            'ds': fc['ds'],
            'yhat': fc[main_col],
            'yhat-lo': fc[lo_col] if lo_col in fc.columns else pd.NA,
            'yhat-hi': fc[hi_col] if hi_col in fc.columns else pd.NA,
        })


def _make_statsforecast(model_cfg: dict, global_cfg) -> StatsForecastModel:
    season_length = (
        model_cfg.get('season_length')
        or global_cfg.get('statsforecast', 'season_length', default=7)
    )
    name = model_cfg.get('name', model_cfg.get('model_class'))
    return StatsForecastModel(
        name=name,
        model_class=model_cfg['model_class'],
        season_length=season_length,
    )


register_factory('statsforecast')(_make_statsforecast)
