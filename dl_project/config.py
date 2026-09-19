# -*- coding: utf-8 -*-
"""
dl_project/config.py — 深度学习项目（NeuralForecast + N-BEATS）

与统计项目、ML 项目并列的第三条轨道：用 N-BEATS 做深度时序预测。
训练集(2017-2020)内做时序 CV，测试集(2021)仅由最终对比脚本使用一次。
"""
from pathlib import Path

# ============ 路径 ============
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
N_WINDOWS = 2          # CV 窗口数（与统计/ML 项目一致，保证可比）
STEP_SIZE = 365        # CV 步长
LEVEL = 95             # 置信区间水平

# ============ N-BEATS 超参数 ============
# input_size：回看窗口。设为一年(365)，既能捕捉年周期，又不至于让训练样本过少。
NB_INPUT_SIZE = 365
# 最大训练步数：CPU 环境下取较小值以控制耗时，可按需调大（GPU 可到 1000+）
NB_MAX_STEPS = 500
NB_BATCH_SIZE = 32
NB_LEARNING_RATE = 0.001
NB_RANDOM_SEED = 1          # 固定随机种子，保证可复现
NB_SCALER = "robust"        # 销量可能有异常值，用 robust 归一化更稳
# 经典 N-BEATS 三栈：identity / trend / seasonality
NB_STACK_TYPES = ["identity", "trend", "seasonality"]
NB_N_BLOCKS = [1, 1, 1]
NB_MLP_UNITS = [[256, 256], [256, 256], [256, 256]]  # 精简版（默认是 [512,512]×3）

# ============ N-HiTS 超参数 ============
NH_INPUT_SIZE = 365
NH_MAX_STEPS = 500
NH_BATCH_SIZE = 32
NH_LEARNING_RATE = 0.001
NH_RANDOM_SEED = 1
NH_SCALER = "robust"
NH_N_BLOCKS = [1, 1, 1]
NH_MLP_UNITS = [[256, 256], [256, 256], [256, 256]]

# ============ TiDE 超参数 ============
TD_INPUT_SIZE = 365
TD_HIDDEN_SIZE = 512
TD_DECODER_OUTPUT_DIM = 32
TD_TEMPORAL_DECODER_DIM = 128
TD_NUM_ENC_LAYERS = 1
TD_NUM_DEC_LAYERS = 1
TD_MAX_STEPS = 500
TD_BATCH_SIZE = 32
TD_LEARNING_RATE = 0.001
TD_RANDOM_SEED = 1
TD_SCALER = "robust"

# ============ DL 参数搜索网格（小规模） ============
# 每模型一个小网格（3 组左右），只调最关键的 1~2 个超参以控制耗时。
# NBEATS 保留现有固定超参（不参与调参）；NHITS / TiDE 各做小规模搜索。
DL_GRID = {
    "NHITS": [
        {"learning_rate": 0.001},
        {"learning_rate": 0.0005},
        {"learning_rate": 0.0001},
    ],
    "TiDE": [
        {"learning_rate": 0.001, "hidden_size": 512},
        {"learning_rate": 0.0005, "hidden_size": 512},
        {"learning_rate": 0.001, "hidden_size": 128},
    ],
}

# ============ 输出文件 ============
DL_CV_REPORT = OUTPUT_DIR / "dl_cv_report.csv"       # 每 SKU N-BEATS 的 CV SMAPE
DL_TEST_PRED = OUTPUT_DIR / "dl_test_predictions.csv"  # 2021 测试期逐日预测（最终对比脚本输入）
