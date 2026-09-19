# -*- coding: utf-8 -*-
"""
ml_project/config.py — ML 项目（MLForecast + LightGBM）独立配置

本项目只保留机器学习模型；统计模型在 time-series-forecast 项目中维护。
训练集(2017-2020)内做时序 CV 调参，测试集(2021)仅由最终对比脚本使用一次。
"""
from pathlib import Path

# ============ 路径 ============
# 指向 Kaggle 分割好的 2017-2020 训练集、2021 测试集（按需修改）
DATA_DIR = Path(
    r"C:/Users/Miracle/Desktop/蟒蛇/kaggle/tutorial/time series/"
    r"playground-series-s3e19_Forecasting Mini-Course Sales/forecast_pipeline_stat_ml/data"
)
TRAIN_FILE = DATA_DIR / "new_train.csv"   # 2017-01-01 ~ 2020-12-31
TEST_FILE = DATA_DIR / "new_test.csv"     # 2021-01-01 ~ 2021-12-31（独立 holdout）

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ 预测 / CV 参数 ============
H = 365                # 预测步长（天）
N_WINDOWS = 2          # CV 窗口数（与统计项目一致，保证可比）
STEP_SIZE = 365        # CV 步长
LEVEL = 95             # 置信区间水平

# ============ 数据列 ============
CAT_COLS = ["country", "store", "product"]    # 类别/静态特征
STATIC_COLS = ["country", "store", "product"]
COUNTRY_CODE = {
    "Argentina": "AR", "Estonia": "EE", "Spain": "ES",
    "Japan": "JP", "Canada": "CA",
}
HOLIDAY_WINDOW = 7

# ============ 精简后的特征配置 ============
# 滞后阶数：短期(1) + 周周期(7) + 月周期(28) + 年周期(365)
# 相比旧版 [1,2,3,7,14,28,30,60,90,180,365] 大幅精简，去掉高冗余/信息量低的阶数
LAGS = [1, 7, 14, 28, 365]

# 滞后变换：只保留信息量高的 RollingMean / RollingStd，去掉与均值高度冗余的 Max/Min
# 结构：{滞后阶数: [(窗口, 类型), ...]}
LAG_TRANSFORM_WINDOWS = {
    7:   [(7, "mean"), (7, "std")],
    14:  [(7, "mean"), (7, "std")],
    28:  [(7, "mean"), (7, "std"), (28, "mean"), (28, "std")],
    365: [(30, "mean"), (30, "std"), (90, "mean"), (90, "std")],
}

# 日期特征开关：全部采用 cyclic(循环) 编码，保证可外推到预测期
# 注意：刻意去掉 year / dayofyear 原始整数，因为它们对预测期(2021)会出现训练未见的取值(泄漏源)
USE_WEEKEND = True     # 是否周末（周几信息已由 is_weekend + 周循环 sin/cos 覆盖）
USE_WEEK_CYCLE = True  # 周循环 sin/cos
USE_MONTH_CYCLE = True # 月循环 sin/cos
USE_YEAR_CYCLE = True  # 年度循环 sin/cos

# ============ LightGBM 精简超参网格 ============
# 只对影响最大的 3 个超参做网格搜索，其余固定为稳健默认值（组合数 = 2*2*2 = 8）
LGB_PARAM_GRID = {
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300, 500],
    "num_leaves": [64, 128],
}
LGB_FIXED = {
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 520,
    "verbosity": -1,
    "device": "cpu",   # 无 GPU 用 cpu；有 GPU 可改 "gpu"
}

# ============ 输出文件 ============
ML_CV_REPORT = OUTPUT_DIR / "ml_cv_report.csv"       # 每 SKU ML 的 CV SMAPE（模型级汇总见 stdout）
ML_BEST_PARAMS = OUTPUT_DIR / "ml_best_params.csv"   # 最优超参
ML_TEST_PRED = OUTPUT_DIR / "ml_test_predictions.csv"  # 2021 测试期逐日预测（最终对比脚本的输入）
