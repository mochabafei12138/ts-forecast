# -*- coding: utf-8 -*-
"""
features.py — 精简后的日期特征与节假日特征

设计原则：
1. 日期特征全部用 cyclic(循环) sin/cos 编码，可外推到预测期(2021)，避免 year/dayofyear
   等原始整数带来的"预测期取值在训练中未见"的泄漏问题。
2. 节假日特征基于日历构造，对未来可知，不构成泄漏。
"""
import numpy as np
import pandas as pd
import holidays as hlib

from config import COUNTRY_CODE, HOLIDAY_WINDOW, H

# ---------------- 日期特征（cyclic 编码） ----------------
# 注：MLForecast 传给 date_features 的是 DatetimeIndex，故统一用 pd.DatetimeIndex 转换，
# 避免 Series 的 .dt 属性与 DatetimeIndex 属性不兼容。
def _idx(dates):
    return pd.DatetimeIndex(dates)

def is_weekend(dates):
    return (_idx(dates).dayofweek >= 5).astype(int)

def week_sin(dates):
    return np.sin(2 * np.pi * _idx(dates).dayofweek / 7)

def week_cos(dates):
    return np.cos(2 * np.pi * _idx(dates).dayofweek / 7)

def month_sin(dates):
    return np.sin(2 * np.pi * _idx(dates).month / 12)

def month_cos(dates):
    return np.cos(2 * np.pi * _idx(dates).month / 12)

def year_sin(dates):
    return np.sin(2 * np.pi * _idx(dates).dayofyear / 365.25)

def year_cos(dates):
    return np.cos(2 * np.pi * _idx(dates).dayofyear / 365.25)


# ---------------- 节假日特征（calendar-based，无泄漏） ----------------
def build_holiday_features(dates, countries, window=HOLIDAY_WINDOW):
    """向量化构造节假日特征，返回 DataFrame(is_holiday, days_to_holiday)。"""
    holiday_cache = {}
    for country, code in COUNTRY_CODE.items():
        for yr in range(dates.dt.year.min(), dates.dt.year.max() + 2):
            try:
                holiday_cache[(country, yr)] = hlib.country_holidays(code, years=yr)
            except Exception:
                holiday_cache[(country, yr)] = {}

    n = len(dates)
    is_holiday = np.zeros(n, dtype=int)
    days_to_holiday = np.zeros(n, dtype=int)
    for i in range(n):
        d = dates.iloc[i].date()
        c = countries.iloc[i]
        hols = holiday_cache.get((c, d.year), {})
        if d in hols:
            is_holiday[i] = 1
            continue
        for delta in range(1, window + 1):
            if (d - pd.Timedelta(days=delta).to_pytimedelta()) in hols or \
               (d + pd.Timedelta(days=delta).to_pytimedelta()) in hols:
                days_to_holiday[i] = delta
                break
    return pd.DataFrame({"is_holiday": is_holiday,
                         "days_to_holiday": days_to_holiday})


def add_holiday_features(df):
    """给训练 df 加上节假日特征列。"""
    feats = build_holiday_features(df["ds"], df["country"])
    out = df.copy()
    out["is_holiday"] = feats["is_holiday"].values
    out["days_to_holiday"] = feats["days_to_holiday"].values
    return out


def build_future_holiday_df(df, h=H):
    """构造未来 h 天每个 SKU 的节假日特征（作为 predict 的动态外生变量）。"""
    rows = []
    for uid, country in df[["unique_id", "country"]].drop_duplicates().values:
        last_date = df[df["unique_id"] == uid]["ds"].max()
        for i in range(1, h + 1):
            rows.append((uid, last_date + pd.Timedelta(days=i), country))
    future = pd.DataFrame(rows, columns=["unique_id", "ds", "country"])
    feats = build_holiday_features(future["ds"], future["country"])
    future["is_holiday"] = feats["is_holiday"].values
    future["days_to_holiday"] = feats["days_to_holiday"].values
    return future
