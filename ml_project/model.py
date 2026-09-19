# -*- coding: utf-8 -*-
"""
model.py — MLForecast + LightGBM：特征组装、时序 CV 调参、测试期逐日预测

只依赖 mlforecast / lightgbm，与统计模型完全解耦。
"""
import itertools

import pandas as pd
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate
from sklearn.preprocessing import LabelEncoder

from config import (
    H, N_WINDOWS, STEP_SIZE, LEVEL, CAT_COLS, STATIC_COLS,
    LAGS, LAG_TRANSFORM_WINDOWS, LGB_PARAM_GRID, LGB_FIXED,
)
import features as F


def build_lag_transforms():
    """由精简配置构造滞后变换：{lag: [RollingMean(w), RollingStd(w), ...]}。"""
    lt = {}
    for lag, windows in LAG_TRANSFORM_WINDOWS.items():
        lt[lag] = [RollingMean(w) if kind == "mean" else RollingStd(w)
                   for w, kind in windows]
    return lt


def build_date_features():
    """按 config 开关组装日期特征（全部 cyclic 编码，可外推）。
    注意：同一特征只添加一次，避免 LGBM 报 "Feature appears more than one time"。"""
    from config import (USE_WEEKEND, USE_WEEK_CYCLE,
                        USE_MONTH_CYCLE, USE_YEAR_CYCLE)
    dfeats = []
    if USE_WEEKEND:
        dfeats.append(F.is_weekend)
    if USE_WEEK_CYCLE:
        dfeats.extend([F.week_sin, F.week_cos])
    if USE_MONTH_CYCLE:
        dfeats.extend([F.month_sin, F.month_cos])
    if USE_YEAR_CYCLE:
        dfeats.extend([F.year_sin, F.year_cos])
    return dfeats


def make_mlforecast(params: dict) -> MLForecast:
    """用给定超参构建 MLForecast 实例（统一入口，保证各阶段特征一致）。"""
    # 网格搜索返回的 pandas 行会把整行统一转成 float（如 n_estimators=500.0），
    # 而 LightGBM 要求整数参数必须为 int，这里强制转回 int 避免 num_iterations 类型报错。
    hp = {**LGB_FIXED, **params}
    for _k in ("n_estimators", "num_leaves", "min_child_samples", "subsample_freq"):
        if _k in hp:
            hp[_k] = int(hp[_k])
    model = lgb.LGBMRegressor(**hp)
    return MLForecast(
        models={"lgb": model},
        freq="D",
        lags=LAGS,
        lag_transforms=build_lag_transforms(),
        date_features=build_date_features(),
        num_threads=4,
    )


def prepare_data(train_df: pd.DataFrame):
    """加节假日特征 + 类别标签编码，返回 (训练数据, 编码器)。"""
    df = F.add_holiday_features(train_df.copy())
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def grid_search_cv(train_hol: pd.DataFrame):
    """在训练集内做时序 CV，网格搜索最优 LightGBM 超参。
    返回 (全组合结果表, 最优参数)。不触碰测试集。"""
    keys = list(LGB_PARAM_GRID.keys())
    values = list(LGB_PARAM_GRID.values())
    results = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        fcst = make_mlforecast(params)
        try:
            cv = fcst.cross_validation(
                train_hol, n_windows=N_WINDOWS, h=H,
                static_features=STATIC_COLS, step_size=STEP_SIZE,
            )
        except Exception as e:
            print(f"  参数 {params} 失败: {e}")
            continue
        eval_df = evaluate(cv, metrics=[smape], models=["lgb"])
        results.append({"mean_smape": float(eval_df["lgb"].mean()), **params})
        print(f"  组合 {params} -> CV mean SMAPE={results[-1]['mean_smape']:.4f}%")
    if not results:
        raise RuntimeError("网格搜索全部组合均失败，请检查特征工程或数据（如滞后阶数超过序列长度）。")
    res = pd.DataFrame(results).sort_values("mean_smape").reset_index(drop=True)
    return res, dict(res.iloc[0][keys])
