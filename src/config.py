# -*- coding: utf-8 -*-
"""
config.py — YAML 配置加载器
支持实验配置通过 inherit 继承 base.yaml，并做深度合并。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / 'configs'


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 优先，嵌套 dict 深合并。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """轻量配置对象，支持点访问与 get()。"""

    def __init__(self, data: dict, source: str | None = None):
        self._data = data
        self.source = source

    def get(self, *keys, default: Any = None) -> Any:
        cur: Any = self._data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            v = self._data[name]
            return Config(v) if isinstance(v, dict) else v
        raise AttributeError(f"Config 中没有键: {name}")

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config(source={self.source})"


def load_config(exp_name: str | None = None,
                configs_dir: str | Path | None = None) -> Config:
    """
    加载实验配置。
    参数 exp_name: 实验配置名（不含 .yaml 后缀），如 'exp_baseline'；
                   或直接传配置文件的绝对路径。
    若不传，则加载 configs/experiments/ 下的第一个实验配置。
    """
    cfg_dir = Path(configs_dir) if configs_dir else CONFIG_DIR
    if exp_name:
        p = Path(exp_name)
        if not p.suffix:
            p = cfg_dir / 'experiments' / f'{exp_name}.yaml'
    else:
        exp_dir = cfg_dir / 'experiments'
        files = sorted(exp_dir.glob('*.yaml'))
        if not files:
            raise FileNotFoundError(
                f"configs/experiments 下没有实验配置文件: {exp_dir}"
            )
        p = files[0]

    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    raw = yaml.safe_load(p.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(f"配置内容不是字典: {p}")

    data: dict = {}
    if raw.get('inherit'):
        base_path = cfg_dir / raw['inherit']
        if not base_path.exists():
            raise FileNotFoundError(f"继承的 base 配置不存在: {base_path}")
        data = _deep_merge(
            yaml.safe_load(base_path.read_text(encoding='utf-8')), raw
        )
    else:
        data = raw
    data.pop('inherit', None)

    # 将相对路径统一解析为相对项目根
    data.setdefault('paths', {})
    data['paths'].setdefault('runs_dir', str(PROJECT_DIR / 'runs'))
    data['paths'].setdefault('db_file', str(PROJECT_DIR / 'experiments.db'))
    return Config(data, source=str(p))


# 便捷：项目根与数据目录
def data_path(rel_path: str) -> Path:
    return PROJECT_DIR / rel_path
