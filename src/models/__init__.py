# -*- coding: utf-8 -*-
"""
models 子包：模型注册表与可插拔模型
"""
from .base import ForecastModel  # noqa: F401
from .registry import (  # noqa: F401
    build_model, build_models, register_factory, available_types,
)
from . import statsforecast_model  # noqa: F401  注册 statsforecast 工厂
from . import prophet_model  # noqa: F401        注册 prophet 工厂
