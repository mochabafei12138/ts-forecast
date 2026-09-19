# -*- coding: utf-8 -*-
"""
blend_predict.py（DL 项目内）— 为跨范式选优生成 DL 选中 SKU 的 2022 全量预测
由 blend_selection.py 以子进程方式调用，不改动现有 main.py。
按每 SKU 的 best_dl_model（NBEATS/NHITS/TiDE）用对应模型预测，并带 95% 区间。
用法：
    python blend_predict.py --skus <json: SKU列表> --out <csv> --data <data目录>
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import model as M
from config import H


def load_best_params(proj_dir: Path):
    fp = proj_dir / "output" / "dl_best_params.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    sku_set = set(json.loads(Path(args.skus).read_text(encoding="utf-8")))

    if args.data:
        train_df = pd.read_csv(Path(args.data) / "train.csv", parse_dates=["ds"])
        test_df = pd.read_csv(Path(args.data) / "test.csv", parse_dates=["ds"])
    else:
        from config import TRAIN_FILE, TEST_FILE
        train_df = pd.read_csv(TRAIN_FILE, parse_dates=["ds"])
        test_df = pd.read_csv(TEST_FILE, parse_dates=["ds"])
    full_df = (pd.concat([train_df, test_df], ignore_index=True)
               .sort_values(["unique_id", "ds"]).reset_index(drop=True))

    proj = Path(__file__).resolve().parent
    report = pd.read_csv(proj / "output" / "dl_cv_report.csv")
    best_model_map = dict(zip(report["unique_id"], report["best_dl_model"]))
    best_params = load_best_params(proj)

    from neuralforecast import NeuralForecast
    from neuralforecast.utils import PredictionIntervals

    parts = []
    for mn in ["NBEATS", "NHITS", "TiDE"]:
        skus = [u for u in sku_set if best_model_map.get(u) == mn]
        if not skus:
            continue
        params = best_params.get(mn, {})
        model = M.DL_MODELS[mn](**params)
        model.trainer_kwargs["enable_progress_bar"] = False
        model.trainer_kwargs["logger"] = False
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(M._long_format(full_df),
               prediction_intervals=PredictionIntervals(n_windows=2))
        p = nf.predict(h=H, level=[95]).reset_index()
        p["ds"] = pd.to_datetime(p["ds"])
        p = p[p["unique_id"].isin(skus)]
        p = p.rename(columns={
            mn: "best_forecast",
            f"{mn}-lo-95": "best_forecast-lo-95",
            f"{mn}-hi-95": "best_forecast-hi-95"})
        keep = ["unique_id", "ds", "best_forecast"]
        if "best_forecast-lo-95" in p.columns:
            keep += ["best_forecast-lo-95", "best_forecast-hi-95"]
        p = p[keep]
        p["best_model"] = mn
        parts.append(p)

    if not parts:
        print("[dl] 选中的 SKU 无对应 DL 模型，跳过。")
        return
    res = pd.concat(parts, ignore_index=True).sort_values(["unique_id", "ds"])
    res.to_csv(args.out, index=False)
    print(f"[dl] 已预测 {res['unique_id'].nunique()} 个 SKU -> {args.out}")


if __name__ == "__main__":
    main()
