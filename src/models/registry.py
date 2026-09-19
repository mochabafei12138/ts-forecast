# -*- coding: utf-8 -*-
"""
registry.py — 模型注册表（可插拔）
新增模型 = 实现 ForecastModel + 注册一个工厂，主流程无需改动。
"""
from __future__ import annotations

from typing import Callable

from .base import ForecastModel

# {type_name: 工厂函数(model_cfg: dict, global_cfg) -> ForecastModel}
_FACTORIES: dict[str, Callable] = {}


def register_factory(type_name: str):
    """注册模型类型的工厂函数。"""
    def deco(fn: Callable):
        _FACTORIES[type_name] = fn
        return fn
    return deco


def build_model(model_cfg: dict, global_cfg) -> ForecastModel:
    """根据一条 YAML 模型定义，构建模型实例。"""
    type_name = model_cfg.get('type')
    if type_name not in _FACTORIES:
        raise ValueError(
            f"未注册的模型类型: {type_name}。可用类型: {list(_FACTORIES)}"
        )
    return _FACTORIES[type_name](model_cfg, global_cfg)


def build_models(exp_cfg) -> list[ForecastModel]:
    """根据实验配置的 models 列表，构建所有模型实例。"""
    models_cfg = exp_cfg.get('models', default=[])
    if not models_cfg:
        raise ValueError("实验配置中 models 为空，请至少注册一个模型")
    return [build_model(mc, exp_cfg) for mc in models_cfg]


def available_types() -> list[str]:
    return list(_FACTORIES)
