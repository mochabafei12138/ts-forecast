"""
streamlit_app.py — 时间序列预测结果交互看板（重构版）
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="时间序列预测看板",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS — 提升整体视觉质感
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1, h2, h3 { font-weight: 600; }
    .stSelectbox label { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 时间序列预测结果看板")
st.caption("多层级 Bottom-Up 汇总 · 全球视角 · 一键探索")
st.divider()

# ============================================================
# 数据路径配置
# ============================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
RUNS_DIR = Path("runs")

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
def _latest_forecast_file():
    """返回 runs/ 下最新 run 的 final_forecast.csv（与版本1框架接轨）。
    若不存在，回退到旧版本2的 output/retrain_final_forecast.csv。"""
    runs = sorted(RUNS_DIR.glob("*/final_forecast.csv"),
                  key=lambda p: p.stat().st_mtime)
    if runs:
        return runs[-1]
    legacy = OUTPUT_DIR / "retrain_final_forecast.csv"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        "未找到最终预测文件。请先运行 time-series-forecast 的 main.py "
        "生成 runs/<run_id>/final_forecast.csv"
    )


# ============================================================
# 国家坐标字典
# ============================================================
COUNTRY_COORDS = {
    "Canada": (56.13, -106.35),
    "United States": (37.09, -95.71),
    "Mexico": (23.63, -102.55),
    "Brazil": (-14.24, -51.93),
    "Argentina": (-38.42, -63.62),
    "Estonia": (58.59, 25.01),
    "Chile": (-35.68, -71.54),
    "Colombia": (4.57, -74.30),
    "Peru": (-9.19, -75.02),
    "United Kingdom": (55.38, -3.44),
    "France": (46.23, 2.21),
    "Germany": (51.17, 10.45),
    "Italy": (41.87, 12.57),
    "Spain": (40.42, -3.70),
    "Portugal": (39.52, -8.39),
    "Netherlands": (52.13, 5.29),
    "Belgium": (50.50, 4.47),
    "Switzerland": (46.62, 7.67),
    "Sweden": (60.47, 8.47),
    "Norway": (60.47, 8.47),
    "Denmark": (56.26, 9.50),
    "Finland": (64.32, 26.81),
    "Poland": (52.33, 19.40),
    "Czech Republic": (49.82, 15.47),
    "Greece": (39.07, 21.82),
    "Turkey": (38.96, 35.24),
    "Russia": (62.10, 96.68),
    "Ukraine": (48.78, 31.51),
    "Romania": (46.61, 25.15),
    "Hungary": (47.16, 19.50),
    "Japan": (36.20, 138.25),
    "South Korea": (37.00, 127.50),
    "China": (35.86, 104.20),
    "India": (20.59, 78.96),
    "Australia": (-25.27, 133.78),
    "South Africa": (-29.00, 24.00),
    "Nigeria": (9.08, 8.68),
    "Kenya": (-0.02, 37.91),
    "Egypt": (26.82, 30.80),
    "Morocco": (31.79, -7.09),
}


# ============================================================
# 数据加载
# ============================================================
@st.cache_data
def load_all_data():
    """加载并合并训练集、测试集和预测结果"""
    train_df = pd.read_csv(TRAIN_FILE)
    train_df["ds"] = pd.to_datetime(train_df["ds"])

    test_df = pd.read_csv(TEST_FILE)
    test_df["ds"] = pd.to_datetime(test_df["ds"])

    forecast_df = pd.read_csv(_latest_forecast_file())
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # 标记数据来源
    train_df["data_type"] = "训练集"
    test_df["data_type"] = "测试集"
    forecast_df["data_type"] = "预测"

    # 对齐列结构
    for df in [train_df, test_df]:
        df["best_forecast"] = np.nan
        df["best_forecast-lo-95"] = np.nan
        df["best_forecast-hi-95"] = np.nan
        df["best_model"] = ""

    forecast_df["y"] = np.nan
    forecast_df["country"] = ""
    forecast_df["store"] = ""
    forecast_df["product"] = ""

    # 从训练/测试集提取元信息
    meta_df = pd.concat([train_df, test_df], ignore_index=True)
    meta_map = meta_df[["unique_id", "country", "store", "product"]].drop_duplicates(
        subset="unique_id"
    )
    meta_dict = {
        row["unique_id"]: {"country": row["country"], "store": row["store"], "product": row["product"]}
        for _, row in meta_map.iterrows()
    }

    for uid, meta in meta_dict.items():
        mask = forecast_df["unique_id"] == uid
        forecast_df.loc[mask, "country"] = meta["country"]
        forecast_df.loc[mask, "store"] = meta["store"]
        forecast_df.loc[mask, "product"] = meta["product"]

    # 合并
    common_cols = [
        "ds", "unique_id", "country", "store", "product",
        "y", "best_forecast", "best_forecast-lo-95", "best_forecast-hi-95",
        "best_model", "data_type",
    ]
    full_df = pd.concat(
        [train_df[common_cols], test_df[common_cols], forecast_df[common_cols]],
        ignore_index=True,
    ).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return full_df, meta_dict


@st.cache_data
def get_country_sku_data(full_df, countries):
    """构建国家地图数据"""
    country_sku_counts = (
        full_df[full_df["data_type"] == "训练集"]
        .groupby("country")["unique_id"]
        .nunique()
        .to_dict()
    )
    rows = []
    for c in countries:
        if c in COUNTRY_COORDS:
            lat, lon = COUNTRY_COORDS[c]
            rows.append({"country": c, "sku_count": country_sku_counts.get(c, 0), "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


# ============================================================
# 加载数据
# ============================================================
with st.spinner("正在加载数据，请稍候..."):
    full_df, meta_dict = load_all_data()

countries = sorted(full_df["country"].dropna().unique().tolist())
country_sku_df = get_country_sku_data(full_df, countries)


# ============================================================
# 侧边栏 — 国家选择
# ============================================================
with st.sidebar:
    st.header("🌍 选择国家")
    selected_country = st.selectbox(
        "选择一个国家以查看详情：",
        [""] + countries,
        index=0,
        key="sidebar_country",
        format_func=lambda x: "— 请选择 —" if x == "" else x,
    )
    st.session_state.selected_country = selected_country if selected_country else None

    st.divider()
    st.caption("提示：选择国家后，下方将展示该国的多层级 Bottom-Up 汇总图。")


# ============================================================
# 全球地图
# ============================================================
st.subheader("🌎 全球国家分布")

hover_texts = country_sku_df.apply(
    lambda r: f"<b>{r['country']}</b><br>SKU 数量: {r['sku_count']}",
    axis=1,
)

fig_map = go.Figure(
    go.Scattergeo(
        lon=country_sku_df["lon"],
        lat=country_sku_df["lat"],
        hovertext=hover_texts,
        hoverinfo="text",
        mode="markers+text",
        marker=dict(
            size=country_sku_df["sku_count"].apply(lambda x: max(10, min(40, x * 2))),
            color=country_sku_df["sku_count"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="SKU 数量", len=0.5),
            line=dict(width=1, color="white"),
        ),
        text=country_sku_df["country"],
        textposition="top center",
        textfont=dict(size=10, color="#2c3e50"),
    )
)

fig_map.update_layout(
    geo=dict(
        showland=True,
        landcolor="rgb(243, 243, 243)",
        countrycolor="rgb(204, 204, 204)",
        showocean=True,
        oceancolor="rgb(230, 240, 255)",
        projection_type="natural earth",
        coastlinecolor="rgb(204, 204, 204)",
    ),
    height=480,
    margin=dict(l=0, r=0, t=0, b=0),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

st.plotly_chart(fig_map, use_container_width=True)
st.divider()


# ============================================================
# 多层级 Bottom-Up 汇总
# ============================================================
st.subheader("📊 多层级 Bottom-Up 汇总")

# 筛选该国数据
country_data = full_df.copy() if not selected_country else full_df[full_df["country"] == selected_country].copy()

# 层级选择
level_options = {
    "国家总览": "country",
    "国家 × 商店": "country_store",
    "国家 × 产品": "country_product",
    "国家 × 商店 × 产品": "country_store_product",
}
selected_level = st.selectbox(
    "选择汇总层级组合：",
    list(level_options.keys()),
    index=0,
    key="level_selector",
)

# 构建子集
subset = country_data.copy()
level_label = selected_level

if selected_level == "国家 × 商店":
    stores = sorted(country_data["store"].dropna().unique())
    sel = st.selectbox("选择商店：", stores, index=0, key="store_lv")
    subset = country_data[country_data["store"] == sel]
    level_label = f"{selected_country} × {sel}"

elif selected_level == "国家 × 产品":
    products = sorted(country_data["product"].dropna().unique())
    sel = st.selectbox("选择产品：", products, index=0, key="prod_lv")
    subset = country_data[country_data["product"] == sel]
    level_label = f"{selected_country} × {sel}"

elif selected_level == "国家 × 商店 × 产品":
    stores = sorted(country_data["store"].dropna().unique())
    sel_store = st.selectbox("选择商店：", stores, index=0, key="store_lv2")
    sub_s = country_data[country_data["store"] == sel_store]
    products = sorted(sub_s["product"].dropna().unique())
    sel_prod = st.selectbox("选择产品：", products, index=0, key="prod_lv2")
    subset = sub_s[sub_s["product"] == sel_prod]
    level_label = f"{selected_country} × {sel_store} × {sel_prod}"

# 按日期汇总
agg_train = (
    subset[subset["data_type"] == "训练集"]
    .groupby("ds")["y"].sum().reset_index()
)
agg_test = (
    subset[subset["data_type"] == "测试集"]
    .groupby("ds")["y"].sum().reset_index()
)
agg_forecast = (
    subset[subset["data_type"] == "预测"]
    .groupby("ds")[["best_forecast", "best_forecast-lo-95", "best_forecast-hi-95"]]
    .sum().reset_index()
)

# 绘图
fig_agg = go.Figure()

if not agg_train.empty:
    fig_agg.add_trace(go.Scatter(
        x=agg_train["ds"], y=agg_train["y"],
        mode="lines", name="训练集",
        line=dict(color="#3b82f6", width=1.5), opacity=0.7,
    ))

if not agg_test.empty:
    fig_agg.add_trace(go.Scatter(
        x=agg_test["ds"], y=agg_test["y"],
        mode="lines+markers", name="测试集真实值",
        line=dict(color="#f59e0b", width=1.5), marker=dict(size=4, symbol="circle"),
    ))

if not agg_forecast.empty:
    # 置信区间（仅最细粒度层级展示）
    if selected_level == "国家 × 商店 × 产品":
        lo = agg_forecast["best_forecast-lo-95"]
        hi = agg_forecast["best_forecast-hi-95"]
        fig_agg.add_trace(go.Scatter(
            x=pd.concat([agg_forecast["ds"], agg_forecast["ds"][::-1]]),
            y=pd.concat([hi, lo[::-1]]),
            fill="toself",
            fillcolor="rgba(34,197,94,0.15)",
            line=dict(width=0),
            name="95% 置信区间",
        ))

    fig_agg.add_trace(go.Scatter(
        x=agg_forecast["ds"], y=agg_forecast["best_forecast"],
        mode="lines", name="预测值",
        line=dict(color="#22c55e", width=2.5),
    ))

# 分界线
if not agg_train.empty:
    train_end = agg_train["ds"].max()
    fig_agg.add_vline(x=train_end, line_dash="dash", line_color="gray", opacity=0.4)
    y_top = max(
        agg_forecast["best_forecast"].max() if not agg_forecast.empty else 0,
        agg_train["y"].max() if not agg_train.empty else 0,
    )
    fig_agg.add_annotation(
        x=train_end, y=y_top * 0.95,
        text="训练 / 预测分界", showarrow=False,
        font=dict(color="gray", size=11),
        xshift=8,
    )

fig_agg.update_layout(
    title=f"**{level_label}** — 日销量汇总",
    xaxis_title="日期",
    yaxis_title="总销量",
    hovermode="x unified",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=30),
)

st.plotly_chart(fig_agg, use_container_width=True)


# ============================================================
# 未选择国家时的默认概览
# ============================================================
# ============================================================
# 单 SKU 预测曲线（跨范式最优模型）
# ============================================================
st.divider()
st.subheader("📈 单 SKU 预测曲线（跨范式最优模型）")

MODEL_COLORS = {
    "dl": "#8b5cf6",
    "Prophet_tuned": "#22c55e",
    "Prophet_plain": "#a3e635",
    "ml": "#f97316",
}

forecast_skus = sorted(full_df[full_df["data_type"] == "预测"]["unique_id"].unique())
sel_sku = st.selectbox("选择一个 SKU 查看其预测曲线：", forecast_skus, key="sku_selector")

sku_data = full_df[full_df["unique_id"] == sel_sku]
train_d = sku_data[sku_data["data_type"] == "训练集"]
test_d = sku_data[sku_data["data_type"] == "测试集"]
pred_d = sku_data[sku_data["data_type"] == "预测"]
model = pred_d["best_model"].iloc[0] if not pred_d.empty else ""

fig_sku = go.Figure()
if not train_d.empty:
    fig_sku.add_trace(go.Scatter(x=train_d["ds"], y=train_d["y"], mode="lines",
                                 name="训练集", line=dict(color="#3b82f6", width=1.4), opacity=0.6))
if not test_d.empty:
    fig_sku.add_trace(go.Scatter(x=test_d["ds"], y=test_d["y"], mode="lines+markers",
                                 name="测试集真实值", line=dict(color="#f59e0b", width=1.5),
                                 marker=dict(size=4)))
if not pred_d.empty:
    lo, hi = pred_d["best_forecast-lo-95"], pred_d["best_forecast-hi-95"]
    fig_sku.add_trace(go.Scatter(x=pd.concat([pred_d["ds"], pred_d["ds"][::-1]]),
                                 y=pd.concat([hi, lo[::-1]]), fill="toself",
                                 fillcolor="rgba(34,197,94,0.12)", line=dict(width=0),
                                 name="95% 置信区间"))
    mcol = MODEL_COLORS.get(model, "#22c55e")
    fig_sku.add_trace(go.Scatter(x=pred_d["ds"], y=pred_d["best_forecast"], mode="lines",
                                 name=f"预测值 · {model}", line=dict(color=mcol, width=2.6)))
if not train_d.empty:
    fig_sku.add_vline(x=train_d["ds"].max(), line_dash="dash", line_color="gray", opacity=0.4)

fig_sku.update_layout(
    title=f"**{sel_sku}** · 最优模型: {model}",
    xaxis_title="日期", yaxis_title="销量", hovermode="x unified", height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=30),
)
st.plotly_chart(fig_sku, use_container_width=True)

# ============================================================
# SKU 最优模型分布
# ============================================================
st.subheader("🏆 SKU 最优模型分布")
model_dist = (
    full_df[full_df["data_type"] == "预测"]
    .groupby("unique_id")["best_model"].last()
    .value_counts()
)
fig_md = go.Figure(go.Pie(
    labels=model_dist.index.tolist(),
    values=model_dist.values.tolist(),
    hole=0.4,
    marker=dict(colors=[MODEL_COLORS.get(k, "#94a3b8") for k in model_dist.index]),
    textinfo="label+value+percent",
))
fig_md.update_layout(height=360, showlegend=True,
                     title="各最优模型覆盖的 SKU 数量")
st.plotly_chart(fig_md, use_container_width=True)

if not selected_country:
    st.info("👈 在左侧边栏选择一个国家，即可查看该国的多层级 Bottom-Up 汇总分析。")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        type_counts = full_df["data_type"].value_counts()
        fig_pie = px.pie(
            values=type_counts.values, names=type_counts.index,
            title="数据量分布（行数）",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        country_bar = (
            full_df[full_df["data_type"] == "训练集"]
            .groupby("country")["unique_id"]
            .nunique()
            .sort_values(ascending=True)
            .reset_index()
        )
        country_bar.columns = ["国家", "SKU 数量"]
        fig_bar = px.bar(
            country_bar.tail(20),
            x="SKU 数量", y="国家",
            orientation="h",
            title="Top 20 国家 SKU 数量",
            color="SKU 数量",
            color_continuous_scale="Viridis",
        )
        fig_bar.update_layout(
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)