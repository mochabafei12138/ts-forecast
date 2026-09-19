# -*- coding: utf-8 -*-
"""
base.py — 统一模型接口
所有模型 wrapper 实现 forecast 与 cv 两个方法，框架据此统一编排、缓存与评估。
"""
from __future__ import annotations

import pandas as pd


class ForecastModel:
    """统一预测模型接口。

    - forecast(): 在给定训练数据上预测未来 h 天（全 SKU）。
      返回 DataFrame，列: [unique_id, ds, yhat, yhat-lo, yhat-hi]
    - cv():       在给定训练数据上做滚动交叉验证。
      返回 (cv_df, cutoffs_map)
        cv_df 列: [unique_id, ds, cutoff, y, yhat]
        cutoffs_map: {unique_id: [cutoff 日期列表]}，供需要固定 cutoffs 的模型使用。
    """

    name: str = 'base'
    # 若为 True，则 cv() 必须由外部提供 cutoffs_map（如 Prophet 用统计模型的 cutoffs）
    needs_cutoffs: bool = False

    def cv(self, train_df: pd.DataFrame, h: int, level: int = 95,
           n_windows: int = 2, step_size: int = 365,
           cutoffs_map: dict | None = None
           ) -> tuple[pd.DataFrame, dict | None]:
        raise NotImplementedError

    def forecast(self, train_df: pd.DataFrame, h: int, level: int = 95
                 ) -> pd.DataFrame:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"
