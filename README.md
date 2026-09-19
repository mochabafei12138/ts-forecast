# 时间序列预测 · 多 SKU 销售预测系统

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B)](https://ts-forecast-ksqvm9vjadiybxxervcu3u.streamlit.app/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ts-forecast-ksqvm9vjadiybxxervcu3u.streamlit.app/)

一套**可持续迭代**的多 SKU 销售时间序列预测流水线：预测 75 个 SKU（5 国 × 多商店 × 多产品）未来 365 天的日销量，在此过程中公平比较**统计模型 / 机器学习 / 深度学习**三类方法，并通过**跨范式 per-SKU 选优**为每个 SKU 挑选最合适的模型上线，做到防泄漏、可复现、可可视化。

数据来自 Kaggle `playground-series-s3e19_Forecasting Mini-Course Sales`。

## 一、系统能力

| 能力 | 说明 |
|------|------|
| 统计预测 | AutoETS / AutoARIMA / AutoTheta / AutoCES / Prophet，per-SKU 选最优做 365 天预测 |
| 机器学习 | MLForecast + LightGBM，精简特征后网格搜索调参 |
| 深度学习 | NeuralForecast + **N-BEATS / N-HiTS / TiDE**，深度家族内 per-SKU 选优 |
| **跨范式选优** | 读三方 per-SKU CV SMAPE，**每个 SKU 跨 stat/ml/dl 取最优模型**，全量重训上线 |
| 严谨评估 | 训练集内时序 CV（n_windows=2）、统一 SMAPE、测试集只碰一次，杜绝数据泄漏 |
| 可视化上线 | Streamlit 看板：全球地图 + 多层级 Bottom-Up 汇总 + **单 SKU 曲线 + 模型分布** |

## 二、数据分割约定

- 训练集：`train.csv` = 2017-01-01 ~ 2020-12-31
- 测试集：`test.csv` = 2021-01-01 ~ 2021-12-31（**独立 holdout，仅最终对比使用一次**）

## 三、目录结构（统计 / ML / DL / 最终对比 / 跨范式 / 看板）

```
├── main.py                 # ① 统计项目入口
├── src/                    #   配置驱动 / 模型可插拔 / 实验追踪 / 断点续跑
├── configs/                #   YAML 实验配置
├── runs/<run_id>/          #   每次运行产物（CV、评估、预测、缓存）
├── streamlit_app.py        #   ⑥ Streamlit 看板
│
├── ml_project/             # ② ML 项目（mlforecast/lightgbm）
│   ├── config.py / features.py / model.py / main.py
│   └── output/             #   ml_cv_report / ml_test_predictions / ml_best_params
│
├── dl_project/             # ③ DL 项目（N-BEATS / N-HiTS / TiDE）
│   ├── config.py           #   三模型超参 + 调参网格 DL_GRID
│   ├── model.py            #   三模型构建 + 网格搜索 + CV + 预测
│   ├── main.py             #   DL 主流程（含家族 per-SKU 选优）
│   ├── blend_predict.py    #   跨范式用（按 best_dl_model 预测 2022）
│   └── output/             #   dl_cv_report / dl_test_predictions / dl_best_params
│
├── final_compare/          # ④ 最终对比（stat vs ml vs dl，一次性 SMAPE 对比）
│   └── main.py
│
├── blend_selection.py      # ⑤ 跨范式 per-SKU 选优（本项目新增的核心一环）
│
├── data/                   # train.csv / test.csv（体积大，不入库）
├── README.md / requirements.txt / run_all.sh / run_all.bat / dt1.bat
```

## 四、完整流程

三条范式各自独立训练，然后做**跨范式 per-SKU 选优**统一上线：

```
                ┌──────────────────────────────────────────┐
                │           数据底座 (data/)                 │
                │  train.csv = 2017~2020   test.csv = 2021  │
                └───────┬──────────────────┬───────────────┘
                        │                  │
   ┌────────────────────▼─────┐  ┌─────────▼──────────────┐
   │①统计项目                  │  │②ML项目                  │
   │·时序CV + 家族per-SKU选优  │  │·精简特征 + 网格调参       │
   │·测试评估 + 全量重训        │  │·全局最优超参             │
   └────────┬─────────────────┘  └─────────┬──────────────┘
            │  model_comparison           │  ml_cv_report
            │  (每SKU家族最优SMAPE)        │  (每SKU SMAPE)
            │                              │
   ┌────────▼─────────────┐  ┌────────────▼──────────┐
   │③DL项目               │  │                       │
   │N-BEATS/N-HiTS/TiDE   │  │                       │
   │·各做小规模调参         │──► dl_cv_report          │
   │·家族per-SKU选优       │  │(每SKU最优DL模型+SMAPE)  │
   └────────┬─────────────┘  │                       │
            │                │                       │
            ▼                ▼                       ▼
   ┌──────────────────────────────────────────────────┐
   │⑤跨范式选优 blend_selection.py                      │
   │ 读三方 per-SKU CV SMAPE（stat家族最优 / ml / dl家族最优）│
   │ 每 SKU 取三者最小 → 跨范式 best_model               │
   │ 按各自范式全量重训 → 预测 2022 → final_forecast.csv  │
   └────────────────────────────┬─────────────────────┘
                                │ final_forecast（2022 上线预测）
                                ▼
                  ┌────────────────────────────┐
                  │⑥ Streamlit 看板             │
                  │ 全球地图 · 多层级汇总         │
                  │ 单SKU曲线(标注所用模型) · 分布│
                  └────────────────────────────┘
```

### 分步流程

**① 统计项目**（根目录）
1. 读数据 → 构建统计模型（AutoETS/AutoARIMA/AutoTheta/AutoCES/Prophet）
2. 时序 CV（n_windows=2, h=365, step=365）——尊重自相关，非随机 K 折
3. 统一 SMAPE 评估 → **per-SKU 在统计家族内选最优**（`model_comparison.csv`）
4. 独立测试集(2021)评估 → 输出 `test_set_predictions.csv`
5. 合并 train+test 全量重训 → 输出上线预测 `final_forecast.csv`
6. 缓存到 `runs/<id>/`，可断点续跑

**② ML 项目**（`ml_project/`）
1. 精简特征（滞后 + 滚动统计 + cyclic 日期特征 + 节假日）
2. 训练集内网格搜索 LightGBM 超参（8 组合，时序 CV 评估，选全局最优）
3. 最优参数重跑 CV → 每 SKU 的 ML SMAPE（`ml_cv_report.csv`）
4. 用训练集训练最优 ML → 生成 2021 逐日预测

**③ DL 项目**（`dl_project/`）
1. 三个深度模型：N-BEATS / N-HiTS / TiDE（均纯单变量）
2. N-BEATS 用固定超参；**N-HiTS、TiDE 各做小规模网格搜索**（`DL_GRID`）
3. 各模型用最优超参跑 CV → 每 SKU 各模型 SMAPE
4. **每 SKU 在 DL 家族内选最优** → `dl_cv_report.csv`（`dl_smape` + `best_dl_model`）
5. 用训练集按 per-SKU 最优 DL 模型 → 生成 2021 逐日预测

**④ 最终对比**（`final_compare/`）
- 加载 stat / ml / dl 的 2021 测试预测 + 测试集真实值，按 (unique_id, ds) 对齐
- 一次性输出 per-SKU / 全局 SMAPE 对比表 + 决策建议（**仅验收，不据此回改模型**）

**⑤ 跨范式选优**（`blend_selection.py`，本项目核心一环）
1. 读三份 per-SKU CV 误差：stat 家族最优（`best_smape`）、ml（`ml_smape`）、dl 家族最优（`dl_smape`）
2. 每 SKU 在 stat / ml / dl 三者里取 SMAPE 最小 → 该 SKU 的跨范式 `best_model`
3. 按选中的范式/模型对 train+test **全量重训**，预测 2022（stat/ml/dl 均带 95% 区间）
4. 输出 `runs/blend_<ts>/final_forecast.csv`（看板读取这份上线结果）

**⑥ Streamlit 看板**
1. 自动读 `runs/` 下最新 `final_forecast.csv`
2. 全球地图 + 多层级 Bottom-Up 汇总（国家 → 商店 → 产品）
3. **单 SKU 预测曲线**（标注该 SKU 用的最优模型，按模型着色）
4. **SKU 最优模型分布**（环形图，看 stat/dl 各覆盖多少 SKU）

## 五、运行顺序

```bash
# 1. 统计项目
python main.py --exp exp_baseline        # 产出 runs/<id>/ 下报告 + final_forecast

# 2. ML 项目
cd ml_project && python main.py          # 产出 output/ml_cv_report.csv 等

# 3. DL 项目（含 N-BEATS/N-HiTS/TiDE 家族选优，较耗时）
cd dl_project && python main.py          # 产出 output/dl_cv_report.csv 等

# 4. 最终对比（stat/ml/dl 三方，验收用）
cd final_compare && python main.py \
  --stat <统计 runs>/<id>/test_set_predictions.csv \
  --ml   ml_project/output/ml_test_predictions.csv \
  --dl   dl_project/output/dl_test_predictions.csv \
  --test data/test.csv

# 5. 跨范式选优 + 上线预测（核心环节）
python blend_selection.py                # 产出 runs/blend_<ts>/final_forecast.csv

# 6. 看板
streamlit run streamlit_app.py
```

## 六、关键约定

- **SMAPE 统一**：统计、ML、DL、最终对比、跨范式选优全部用 `utilsforecast.losses.smape`，公式一致、0~1 尺度、数值直接可比。
- **测试集只用一次**：各项目只在训练集内调参选优；测试集(2021)只在 `final_compare` 中一次性用于最终确认与展示，**不据此回改模型**。
- **per-SKU 选优**：每个 SKU 用最适合自己的模型——统计家族内选、DL 家族内选、最后**跨 stat/ml/dl 再选**。
- **防泄漏**：ML 日期特征用 cyclic sin/cos 编码，去掉 `year`/`dayofyear` 原始整数；跨范式选优基于训练集 CV，绝不用测试集挑模型。
