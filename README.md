# 时间序列预测 · 多 SKU 销售预测系统

一套**可持续迭代**的多 SKU 销售时间序列预测流水线：预测 75 个 SKU（5 国 × 多商店 × 多产品）未来 365 天的日销量，在此过程中公平比较**统计模型 / 机器学习 / 深度学习**三类方法，做到防泄漏、可复现、可可视化。

数据来自 Kaggle `playground-series-s3e19_Forecasting Mini-Course Sales`。

## 一、系统能力

| 能力 | 说明 |
|------|------|
| 统计预测 | AutoETS / AutoARIMA / AutoTheta / AutoCES / Prophet，per-SKU 选最优做 365 天预测 |
| 机器学习 | MLForecast + LightGBM，精简特征后同口径预测 |
| 深度学习 | NeuralForecast + N-BEATS，深度时序模型（本项目的第三条轨道） |
| 严谨评估 | 训练集内时序 CV（n_windows=2）、统一 SMAPE、测试集只碰一次，杜绝数据泄漏 |
| 可视化上线 | Streamlit 看板：全球地图 + 多层级 Bottom-Up 汇总（国家 → 商店 → 产品） |

## 二、数据分割约定

- 训练集：`train.csv` = 2017-01-01 ~ 2020-12-31
- 测试集：`test.csv` = 2021-01-01 ~ 2021-12-31（**独立 holdout，仅最终对比使用一次**）

## 三、目录结构（四个项目 + 看板）

```
├── time-series-forecast/    # ① 统计项目（版本1框架）
│   ├── src/                 #   配置驱动 / 模型可插拔 / 实验追踪 / 断点续跑
│   ├── configs/             #   YAML 实验配置
│   ├── runs/<run_id>/       #   每次运行产物（CV、评估、预测、缓存）
│   ├── streamlit_app.py     #   ⑤ Streamlit 看板
│   └── main.py              #   运行入口
│
├── ml_project/              # ② ML 项目（只保留 mlforecast/lightgbm）
│   ├── config.py            #   精简特征 + CV 参数配置
│   ├── features.py          #   日期特征(cyclic) + 节假日特征
│   ├── model.py             #   MLForecast 组装、CV 调参、预测
│   ├── main.py              #   ML 主流程
│   └── output/              #   ml_cv_report / ml_test_predictions / ml_best_params
│
├── dl_project/              # ③ DL 项目（NeuralForecast + N-BEATS）
│   ├── config.py            #   N-BEATS 超参 + CV 参数
│   ├── model.py             #   N-BEATS 训练、CV、预测
│   ├── main.py              #   DL 主流程
│   └── output/              #   dl_cv_report / dl_test_predictions
│
└── final_compare/           # ④ 最终对比
    └── main.py              #   加载 stat+ml+dl+test，一次性 SMAPE 对比
```

## 四、完整流程

三个模型各自独立跑，最后统一对比：

```
                ┌──────────────────────────────────────────┐
                │           数据底座 (data/)                 │
                │  train.csv = 2017~2020   test.csv = 2021  │
                └───────┬──────────────────┬───────────────┘
                        │                  │
   ┌────────────────────▼─────┐  ┌─────────▼──────────────┐
   │①统计项目                  │  │②ML项目                  │
   │·时序CV + per-SKU选优      │  │·精简特征(~29)           │
   │·测试评估 + 全量重训        │  │·网格搜索调参             │
   └────────┬─────────────────┘  └─────────┬──────────────┘
            │  test_set_predictions        │  ml_test_predictions
            │  (2021逐日)                  │  (2021逐日)
            │  final_forecast (2022上线)─►⑤Streamlit看板
            │                              │
   ┌────────▼─────────────┐  ┌────────────▼──────────┐
   │③DL项目(N-BEATS)       │  │                       │
   │·时序CV                 │  │                       │
   │·深度时序预测            │──► dl_test_predictions  │
   └────────┬─────────────┘  │   (2021逐日)           │
            │                │                       │
            └─────────┐  ┌───┴──────────┐            │
                      ▼  ▼              ▼            │
              ┌──────────────────────────────────┐   │
              │④最终对比 final_compare            │   │
              │ stat + ml + dl + test 对齐        │──┘
              │ 2021一次性 SMAPE 对比 + 决策建议   │
              └──────────────────────────────────┘
```

### 分步流程

**① 统计项目**（`time-series-forecast/`）
1. 读数据 → 构建统计模型（注册表可插拔）
2. 时序 CV（n_windows=2, h=365, step=365）——尊重自相关，非随机 K 折
3. 统一 SMAPE 评估 → **per-SKU 选最优模型**
4. 独立测试集(2021)评估 → 输出 2021 逐日预测 `test_set_predictions.csv`
5. 合并 train+test 全量重训 → 输出上线预测 `final_forecast.csv`
6. 全程写入 SQLite 实验库、缓存到 `runs/<id>/`，可断点续跑

**② ML 项目**（`ml_project/`）
1. 精简特征（滞后 `[1,7,14,28,365]` + 滚动 Mean/Std + cyclic 日期特征 + 节假日）
2. 训练集内网格搜索 LightGBM 超参（8 组合，时序 CV 评估）
3. 最优参数重跑 CV → 每 SKU 的 ML SMAPE
4. 用训练集训练最优 ML → 生成 2021 逐日预测 `ml_test_predictions.csv`

**③ DL 项目**（`dl_project/`）
1. N-BEATS：多层 MLP 堆叠分解趋势/季节性的深度时序模型
2. 训练集内时序 CV（h/step/n_windows 与统计/ML 一致）
3. 用训练集训练 N-BEATS → 生成 2021 逐日预测 `dl_test_predictions.csv`

**④ 最终对比**（`final_compare/`）
1. 加载 stat + ml（+dl）预测 + 测试集真实值
2. 三者/四者按 (unique_id, ds) 对齐
3. 一次性输出 per-SKU / 全局 SMAPE 对比表 + 决策建议

**⑤ Streamlit 看板**
1. 自动读 `runs/` 下最新 run 的 `final_forecast.csv`
2. 全球地图（国家分布、SKU 数量）
3. 多层级 Bottom-Up 汇总：选国家 → 选商店 → 选产品 → 展示销量曲线

## 五、运行顺序

```bash
# 1. 统计项目
cd time-series-forecast
python main.py --exp exp_baseline        # 产出 runs/<id>/final_forecast.csv

# 2. ML 项目
cd ml_project
pip install -r requirements.txt
python main.py                            # 产出 output/ml_test_predictions.csv

# 3. DL 项目（可选，深度时序）
cd dl_project
pip install -r requirements.txt
python main.py                            # 产出 output/dl_test_predictions.csv

# 4. 最终对比（stat/ml/dl 三方；不带 --dl 则仅 stat/ml）
cd final_compare
python main.py \
  --stat <统计 runs>/<id>/test_set_predictions.csv \
  --ml   ml_project/output/ml_test_predictions.csv \
  --dl   dl_project/output/dl_test_predictions.csv \
  --test <数据>/new_test.csv

# 5. 看板
cd time-series-forecast
streamlit run streamlit_app.py
```

## 六、关键约定

- **SMAPE 三处/四处统一**：统计、ML、DL、最终对比全部用 `utilsforecast.losses.smape`，公式一致、0~1 尺度、数值直接可比。
- **测试集只用一次**：各项目只在训练集内调参选优；测试集(2021)只在 `final_compare` 中一次性用于最终确认与展示，**不据此回改模型**。
- **per-SKU 选优**：每个 SKU 用最适合自己的模型（统计内部也在选）。
- **防泄漏**：ML 日期特征用 cyclic sin/cos 编码，去掉 `year`/`dayofyear` 原始整数。
