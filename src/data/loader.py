# -*- coding: utf-8 -*-
"""
loader.py — 数据加载与切分策略
将"训练窗口 / 评估窗口 / 上线预测目标"显式化，避免数据更新时误切分。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import PROJECT_DIR


@dataclass
class DatasetInfo:
    """一次加载的数据切分元信息，用于实验追踪。"""
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    train_ds_range: tuple[str, str]
    test_ds_range: tuple[str, str]
    n_sku: int
    n_country: int
    n_store: int
    n_product: int

    def to_dict(self) -> dict:
        return {
            'train_shape': list(self.train_shape),
            'test_shape': list(self.test_shape),
            'train_ds_range': list(self.train_ds_range),
            'test_ds_range': list(self.test_ds_range),
            'n_sku': self.n_sku,
            'n_country': self.n_country,
            'n_store': self.n_store,
            'n_product': self.n_product,
        }


def load_data(train_file: str | None = None,
              test_file: str | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame, DatasetInfo]:
    """
    加载训练集与测试集。
    约定：训练集不含评估窗口，测试集为独立 holdout（含标签）。
    返回 (train_df, test_df, info)。
    """
    train_file = train_file or str(PROJECT_DIR / 'data' / 'train.csv')
    test_file = test_file or str(PROJECT_DIR / 'data' / 'test.csv')

    train_df = pd.read_csv(train_file, parse_dates=['ds'])
    test_df = pd.read_csv(test_file, parse_dates=['ds'])

    # 排序保证序列一致
    train_df = train_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    test_df = test_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    info = DatasetInfo(
        train_shape=train_df.shape,
        test_shape=test_df.shape,
        train_ds_range=(str(train_df['ds'].min().date()),
                        str(train_df['ds'].max().date())),
        test_ds_range=(str(test_df['ds'].min().date()),
                       str(test_df['ds'].max().date())),
        n_sku=train_df['unique_id'].nunique(),
        n_country=train_df['country'].nunique(),
        n_store=train_df['store'].nunique(),
        n_product=train_df['product'].nunique(),
    )

    print(f"训练集: {train_df.shape}, 日期 {info.train_ds_range[0]} ~ {info.train_ds_range[1]}, "
          f"SKU={info.n_sku}")
    print(f"测试集: {test_df.shape}, 日期 {info.test_ds_range[0]} ~ {info.test_ds_range[1]}")

    return train_df, test_df, info


def get_all_sku_ids(train_df: pd.DataFrame) -> list[str]:
    return list(train_df['unique_id'].unique())
