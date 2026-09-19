# -*- coding: utf-8 -*-
"""
experiment.py — SQLite 实验追踪
记录每次运行的配置快照、git 版本、指标、模型分布，支撑跨实验对比与回溯。
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


class ExperimentTracker:
    def __init__(self, db_file: str | Path, project_dir: str | Path | None = None):
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.project_dir = Path(project_dir) if project_dir else None
        self._conn = sqlite3.connect(str(self.db_file))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id          TEXT PRIMARY KEY,
                experiment      TEXT,
                config_json     TEXT,
                git_commit      TEXT,
                created_at      TEXT,
                finished_at     TEXT,
                status          TEXT,
                total_time_sec  REAL,
                note            TEXT
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS run_metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT,
                unique_id   TEXT,
                best_model  TEXT,
                cv_smape    REAL,
                test_smape  REAL,
                test_n      INTEGER
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_summary (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT,
                model       TEXT,
                mean_smape  REAL,
                n_sku       INTEGER,
                stage       TEXT
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS best_model_dist (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id  TEXT,
                model   TEXT,
                count   INTEGER
            )""")
        self._conn.commit()

    # ---------------- git ----------------
    @staticmethod
    def get_git_commit(project_dir: str | Path) -> str:
        try:
            out = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=str(project_dir), capture_output=True, text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else ''
        except Exception:
            return ''

    # ---------------- run 生命周期 ----------------
    def start_run(self, experiment: str, config: dict,
                  note: str = '', git_commit: str | None = None) -> str:
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        commit = git_commit
        if commit is None and self.project_dir is not None:
            commit = self.get_git_commit(self.project_dir)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO runs (run_id, experiment, config_json, git_commit, "
            "created_at, status, note) VALUES (?,?,?,?,?,?,?)",
            (run_id, experiment, json.dumps(config), commit or '',
             datetime.now().isoformat(), 'running', note),
        )
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, status: str, total_time_sec: float) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE runs SET status=?, finished_at=?, total_time_sec=? "
            "WHERE run_id=?",
            (status, datetime.now().isoformat(), total_time_sec, run_id),
        )
        self._conn.commit()

    def mark_run_failed(self, run_id: str, note: str = '') -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE runs SET status='failed', finished_at=?, note=? WHERE run_id=?",
            (datetime.now().isoformat(), note, run_id),
        )
        self._conn.commit()

    # ---------------- 结果写入 ----------------
    def save_metrics(self, run_id: str, metrics_df: pd.DataFrame) -> None:
        if metrics_df is None or metrics_df.empty:
            return
        # 幂等：先清空该 run 的旧记录，避免 resume/重跑累积重复
        self._conn.execute("DELETE FROM run_metrics WHERE run_id=?", (run_id,))
        self._conn.commit()
        rows = []
        for _, r in metrics_df.iterrows():
            rows.append((
                run_id, r.get('unique_id'), r.get('best_model'),
                r.get('cv_smape'), r.get('test_smape'), r.get('test_n'),
            ))
        self._conn.executemany(
            "INSERT INTO run_metrics (run_id, unique_id, best_model, "
            "cv_smape, test_smape, test_n) VALUES (?,?,?,?,?,?)", rows)
        self._conn.commit()

    def save_model_summary(self, run_id: str, summary_df: pd.DataFrame,
                           stage: str = 'cv') -> None:
        if summary_df is None or summary_df.empty:
            return
        self._conn.execute(
            "DELETE FROM model_summary WHERE run_id=? AND stage=?",
            (run_id, stage))
        self._conn.commit()
        rows = []
        for _, r in summary_df.iterrows():
            rows.append((run_id, r.get('model'), r.get('mean_smape'),
                         r.get('n_sku'), stage))
        self._conn.executemany(
            "INSERT INTO model_summary (run_id, model, mean_smape, n_sku, stage) "
            "VALUES (?,?,?,?,?)", rows)
        self._conn.commit()

    def save_best_model_dist(self, run_id: str, dist: dict) -> None:
        self._conn.execute(
            "DELETE FROM best_model_dist WHERE run_id=?", (run_id,))
        self._conn.commit()
        rows = [(run_id, m, c) for m, c in dist.items()]
        self._conn.executemany(
            "INSERT INTO best_model_dist (run_id, model, count) VALUES (?,?,?)",
            rows)
        self._conn.commit()

    # ---------------- 查询 ----------------
    def list_runs(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT run_id, experiment, git_commit, created_at, status, "
            "total_time_sec, note FROM runs ORDER BY created_at DESC",
            self._conn)

    def get_run_metrics(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT unique_id, best_model, cv_smape, test_smape, test_n "
            "FROM run_metrics WHERE run_id=? ORDER BY cv_smape",
            self._conn, params=(run_id,))

    def get_run_summary(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT model, mean_smape, n_sku, stage FROM model_summary "
            "WHERE run_id=?", self._conn, params=(run_id,))

    def get_run_config(self, run_id: str) -> dict:
        row = self._conn.execute(
            "SELECT config_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return {}
        return json.loads(row['config_json'])

    def close(self) -> None:
        self._conn.close()
