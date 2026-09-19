# -*- coding: utf-8 -*-
"""
model.py — NeuralForecast 多深度模型：N-BEATS / N-HiTS / TiDE

每个模型都支持小规模参数搜索（grid_search_cv），并在 DL 家族内做 per-SKU 选优。
所有模型只用 unique_id / ds / y（纯单变量），与统计/ML 项目解耦。
"""
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, TiDE
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate

from config import (
    H, N_WINDOWS, STEP_SIZE,
    NB_INPUT_SIZE, NB_MAX_STEPS, NB_BATCH_SIZE, NB_LEARNING_RATE,
    NB_RANDOM_SEED, NB_SCALER, NB_STACK_TYPES, NB_N_BLOCKS, NB_MLP_UNITS,
    NH_INPUT_SIZE, NH_MAX_STEPS, NH_BATCH_SIZE, NH_LEARNING_RATE,
    NH_RANDOM_SEED, NH_SCALER, NH_N_BLOCKS, NH_MLP_UNITS,
    TD_INPUT_SIZE, TD_HIDDEN_SIZE, TD_DECODER_OUTPUT_DIM, TD_TEMPORAL_DECODER_DIM,
    TD_NUM_ENC_LAYERS, TD_NUM_DEC_LAYERS, TD_MAX_STEPS, TD_BATCH_SIZE,
    TD_LEARNING_RATE, TD_RANDOM_SEED, TD_SCALER,
)


def build_nbeats(**overrides) -> NBEATS:
    """构建 N-BEATS 模型（固定超参，不参与调参）。"""
    params = dict(
        h=H, input_size=NB_INPUT_SIZE,
        stack_types=NB_STACK_TYPES, n_blocks=NB_N_BLOCKS, mlp_units=NB_MLP_UNITS,
        max_steps=NB_MAX_STEPS, batch_size=NB_BATCH_SIZE,
        learning_rate=NB_LEARNING_RATE, random_seed=NB_RANDOM_SEED,
        scaler_type=NB_SCALER,
    )
    params.update(overrides)
    model = NBEATS(**params)
    # 禁用 Lightning rich 进度条：中文 Windows(GBK) 控制台渲染特殊字符会崩溃
    model.trainer_kwargs["enable_progress_bar"] = False
    model.trainer_kwargs["logger"] = False
    return model


def build_nhits(**overrides) -> NHITS:
    """构建 N-HiTS 模型。overrides 可覆盖超参（用于调参）。"""
    params = dict(
        h=H, input_size=NH_INPUT_SIZE,
        n_blocks=NH_N_BLOCKS, mlp_units=NH_MLP_UNITS,
        max_steps=NH_MAX_STEPS, batch_size=NH_BATCH_SIZE,
        learning_rate=NH_LEARNING_RATE, random_seed=NH_RANDOM_SEED,
        scaler_type=NH_SCALER,
    )
    params.update(overrides)
    model = NHITS(**params)
    model.trainer_kwargs["enable_progress_bar"] = False
    model.trainer_kwargs["logger"] = False
    return model


def build_tide(**overrides) -> TiDE:
    """构建 TiDE 模型。overrides 可覆盖超参（用于调参）。"""
    params = dict(
        h=H, input_size=TD_INPUT_SIZE, hidden_size=TD_HIDDEN_SIZE,
        decoder_output_dim=TD_DECODER_OUTPUT_DIM,
        temporal_decoder_dim=TD_TEMPORAL_DECODER_DIM,
        num_encoder_layers=TD_NUM_ENC_LAYERS,
        num_decoder_layers=TD_NUM_DEC_LAYERS,
        max_steps=TD_MAX_STEPS, batch_size=TD_BATCH_SIZE,
        learning_rate=TD_LEARNING_RATE, random_seed=TD_RANDOM_SEED,
        scaler_type=TD_SCALER,
    )
    params.update(overrides)
    model = TiDE(**params)
    model.trainer_kwargs["enable_progress_bar"] = False
    model.trainer_kwargs["logger"] = False
    return model


# 深度学习模型注册表
DL_MODELS = {
    "NBEATS": build_nbeats,
    "NHITS": build_nhits,
    "TiDE": build_tide,
}


def _long_format(train_df: pd.DataFrame) -> pd.DataFrame:
    """深度模型只需要长格式的 unique_id/ds/y。"""
    return train_df[["unique_id", "ds", "y"]].copy()


def run_cv(train_df: pd.DataFrame, model=None, h=None) -> pd.DataFrame:
    """对单个模型做时序 CV，返回含 <模型名> 预测列与真实值 y 的 CV 结果。
    窗口设置(h/step/n_windows)与统计、ML 项目一致，保证可比。"""
    h = h or H
    nf = NeuralForecast(models=[model], freq="D")
    cv = nf.cross_validation(
        df=_long_format(train_df), h=h,
        n_windows=N_WINDOWS, step_size=STEP_SIZE, verbose=False,
    )
    return cv.reset_index()


def grid_search_cv(train_df: pd.DataFrame, model_name: str, grid, h=None):
    """对某模型的一组网格超参跑 CV，返回 (结果表, 全局最优超参)。
    grid: list[dict]，每个 dict 是一组超参。"""
    h = h or H
    rows = []
    for params in grid:
        model = DL_MODELS[model_name](**params)
        cv = run_cv(train_df, model=model, h=h)
        ev = evaluate(cv, metrics=[smape], models=[model_name])
        rows.append({"mean_smape": float(ev[model_name].mean()), **params})
    res = pd.DataFrame(rows).sort_values("mean_smape").reset_index(drop=True)
    # 把从 DataFrame 取出的 numpy 标量转回纯 Python 类型（numpy.float64 会破坏
    # TiDE 等模型的 LayerNorm(output_dim) 等需要 int 的地方）
    best = {}
    for k in grid[0]:
        v = res.iloc[0][k]
        if isinstance(v, (np.integer, np.floating)):
            v = v.item()
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        best[k] = v
    return res, best


def forecast(train_df: pd.DataFrame, model=None, h=None, level=None) -> pd.DataFrame:
    """用训练集训练单个模型，预测未来 h 天，返回逐日预测（含可选区间）。"""
    h = h or H
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=_long_format(train_df))
    kw = {"level": level} if level else {}
    preds = nf.predict(h=h, **kw).reset_index()
    preds["ds"] = pd.to_datetime(preds["ds"])
    return preds
