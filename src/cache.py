# -*- coding: utf-8 -*-
"""
cache.py — 断点续跑缓存
同一 run 内，模型级 cv / forecast 结果按 key 缓存为 parquet。
进程中断后通过 --resume <run_id> 恢复，可跳过已完成阶段。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class CacheStore:
    def __init__(self, run_dir: str | Path):
        self.base = Path(run_dir) / 'cache'
        self.base.mkdir(parents=True, exist_ok=True)

    def _file(self, group: str, key: str) -> Path:
        d = self.base / group
        d.mkdir(parents=True, exist_ok=True)
        # key 中可能含斜杠等，做安全化
        safe = str(key).replace('/', '_').replace('\\', '_')
        return d / f'{safe}.parquet'

    def exists(self, group: str, key: str) -> bool:
        return self._file(group, key).exists()

    def save(self, group: str, key: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df.to_parquet(self._file(group, key), index=False)

    def load(self, group: str, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._file(group, key))

    def save_json(self, group: str, key: str, obj) -> None:
        import json
        d = self.base / group
        d.mkdir(parents=True, exist_ok=True)
        safe = str(key).replace('/', '_').replace('\\', '_')
        (d / f'{safe}.json').write_text(json.dumps(obj), encoding='utf-8')

    def load_json(self, group: str, key: str):
        import json
        safe = str(key).replace('/', '_').replace('\\', '_')
        f = self.base / group / f'{safe}.json'
        if not f.exists():
            return None
        return json.loads(f.read_text(encoding='utf-8'))
