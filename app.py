"""台北實價登錄分析儀表板 (Streamlit)。

兩頁：資料分布概況、房價試算。
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from src import model as model_mod
from src import data_pipeline as dp

# 視覺主題
PRIMARY_COLOR = "#1f4e79"
SURFACE_COLOR = "#f5f5f7"
TEXT_COLOR = "#2c3e50"
CHART_PALETTE = "Blues"
FONT_STACK = (
    "'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', "
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
)

st.set_page_config(
    page_title="台北實價登錄分析",
    layout="wide",
)

DB_PATH = dp.DEFAULT_DB_PATH


def inject_styles() -> None:
    """注入字型與排版微調。"""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;600;700&display=swap');

        html, body, .stApp, .stMarkdown, button, input, select, textarea {{
            font-family: {FONT_STACK};
            color: {TEXT_COLOR};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: {FONT_STACK};
            color: {TEXT_COLOR};
            font-weight: 600;
        }}
        h2 {{
            font-size: 1.45rem;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 0.4rem;
            margin-bottom: 1.2rem;
        }}
        h3 {{
            font-size: 1.05rem;
            margin-top: 1.6rem;
            margin-bottom: 0.4rem;
            color: #4a5568;
            font-weight: 500;
        }}
        [data-testid="stMetric"] {{
            background-color: {SURFACE_COLOR};
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #ebebee;
        }}
        [data-testid="stMetricLabel"] {{
            color: #6b7280;
            font-size: 0.82rem;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #fafafa;
        }}
        section[data-testid="stSidebar"] h3 {{
            margin-top: 0.5rem;
        }}
        /* Sidebar radio 圓圈未選取時邊框加深，避免與背景融合 */
        div[role="radiogroup"] label > div:first-child {{
            border: 1.5px solid #888 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="載入資料中…")
def load_data() -> pd.DataFrame:
    """從 SQLite 讀取清整資料；若不存在則先跑 pipeline。"""
    if not os.path.exists(DB_PATH):
        dp.run_pipeline(DB_PATH)
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql("SELECT * FROM transactions", con)
    return df


@st.cache_resource(show_spinner="載入模型中…")
def load_model_resources():
    """載入訓練好的模型與 metadata；若不存在則訓練。"""
    if not os.path.exists(model_mod.MODEL_PATH):
        model_mod.train(DB_PATH)
    return model_mod.load_model()


def style_chart(fig, height: int = 400):
    """套用統一視覺：透明背景、淡灰格線、品牌字型。"""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=12, color=TEXT_COLOR),
        margin=dict(t=20, b=40, l=10, r=10),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor="rgba(0,0,0,0.15)", tickcolor="rgba(0,0,0,0.15)",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
        linecolor="rgba(0,0,0,0)",
    )
    return fig


def render_sidebar() -> str:
    """側邊欄頁面導覽。"""
    st.sidebar.markdown("### 台北實價登錄分析")
    page = st.sidebar.radio("選擇頁面", ["資料分布概況", "房價試算"], label_visibility="collapsed")
    return page


def render_topbar(df: pd.DataFrame) -> None:
    """頁面頂端資料來源橫幅。"""
    st.markdown(
        f"""
        <div style="background-color: {SURFACE_COLOR}; padding: 10px 16px;
                     border-radius: 6px; border-left: 3px solid {PRIMARY_COLOR};
                     margin-bottom: 22px; font-size: 0.85rem; color: #4a5568;">
            資料來源：內政部不動產交易實價登錄 · 台北市 · 2025Q2–2026Q1 ·
            <strong style="color: {TEXT_COLOR};">{len(df):,}</strong> 筆有效樣本
        </div>
        """,
        unsafe_allow_html=True,
    )


def district_order_by_median(df: pd.DataFrame) -> list[str]:
    """全市資料計算各區單價中位數，回傳由高到低的行政區清單。"""
    return (
        df.groupby("行政區")["單價(萬/坪)"].median()
        .sort_values(ascending=False).index.tolist()
    )


def page_explore(df: pd.DataFrame) -> None:
    """資料分布概況頁面。"""
    st.markdown("## 資料分布概況")
    st.caption("依行政區檢視台北市住宅單價分布")

    sorted_districts = district_order_by_median(df)
    selected = st.multiselect(
        "行政區",
        options=sorted_districts,
        default=sorted_districts,
    )

    if not selected:
        st.warning("請至少選一個行政區。")
        return

    sub = df[df["行政區"].isin(selected)]

    # KPI：與全市比較
    avg_sub = sub["單價(萬/坪)"].mean()
    avg_all = df["單價(萬/坪)"].mean()
    med_sub = sub["單價(萬/坪)"].median()
    med_all = df["單價(萬/坪)"].median()
    avg_delta = (avg_sub - avg_all) / avg_all * 100
    med_delta = (med_sub - med_all) / med_all * 100
    pct_of_total = len(sub) / len(df) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "平均單價（萬/坪）", f"{avg_sub:.2f}",
        delta=f"{avg_delta:+.1f}% vs 全市", delta_color="off",
    )
    c2.metric(
        "中位數（萬/坪）", f"{med_sub:.2f}",
        delta=f"{med_delta:+.1f}% vs 全市", delta_color="off",
    )
    c3.metric(
        "樣本數", f"{len(sub):,}",
        delta=f"佔全市 {pct_of_total:.1f}%", delta_color="off",
    )

    st.markdown("### 單價分布")
    fig_hist = px.histogram(
        sub, x="單價(萬/坪)", nbins=40,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig_hist.update_layout(
        bargap=0.05,
        yaxis_title="交易筆數",
        xaxis_title="單價（萬/坪）",
    )
    style_chart(fig_hist, height=340)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### 各行政區單價排名（中位數）")
    st.caption("此圖顯示全市 12 區排名，不受上方行政區篩選影響")
    # 永遠顯示全市 12 區，洞察文字也以全資料為準
    agg_d = (
        df.groupby("行政區")["單價(萬/坪)"]
        .agg(中位數="median", 樣本數="count")
        .reset_index()
        .sort_values("中位數", ascending=False)
    )
    agg_d["標註"] = agg_d.apply(
        lambda r: f"{r['中位數']:.1f}<br>n={int(r['樣本數']):,}", axis=1,
    )
    if len(agg_d) >= 2:
        high_d, high_v = agg_d.iloc[0]["行政區"], agg_d.iloc[0]["中位數"]
        low_d, low_v = agg_d.iloc[-1]["行政區"], agg_d.iloc[-1]["中位數"]
        st.info(
            f"**{high_d}** 單價最高（約 {high_v:.1f} 萬/坪），"
            f"**{low_d}** 最低（約 {low_v:.1f} 萬/坪），"
            f"差距約 {high_v - low_v:.1f} 萬/坪。"
        )
    else:
        only = agg_d.iloc[0]
        st.info(f"**{only['行政區']}** 中位數單價約 {only['中位數']:.1f} 萬/坪。")

    fig_dist = px.bar(
        agg_d, x="行政區", y="中位數",
        color="中位數", color_continuous_scale=CHART_PALETTE,
        text="標註",
        category_orders={"行政區": agg_d["行政區"].tolist()},
    )
    fig_dist.update_traces(
        textposition="outside",
        cliponaxis=False,
        customdata=[[c] for c in agg_d["樣本數"].tolist()],
        hovertemplate=(
            "行政區：%{x}<br>"
            "中位數單價：%{y:,.1f} 萬/坪<br>"
            "樣本數：%{customdata[0]:,} 筆"
            "<extra></extra>"
        ),
    )
    fig_dist.update_layout(
        coloraxis_showscale=False,
        yaxis_title="中位數單價（萬/坪）",
        xaxis_title=None,
    )
    style_chart(fig_dist, height=440)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### 不同坪數的單價比較")
    st.caption(
        "全市資料計算，不受上方行政區篩選影響　·　"
        "坪數均為權狀坪數（含公設），與內政部實價登錄、各大房仲網站口徑一致"
    )

    size_bins = [5, 15, 30, 50, 100, float("inf")]
    size_labels = ["5-15坪", "15-30坪", "30-50坪", "50-100坪", "100坪以上"]
    # 永遠顯示全市坪數結構，洞察文字也以全資料為準
    df_with_bin = df.copy()
    df_with_bin["坪數區間"] = pd.cut(
        df_with_bin["總坪數"],
        bins=size_bins, labels=size_labels,
        right=False, include_lowest=True,
    )
    agg_b = (
        df_with_bin.dropna(subset=["坪數區間"])
        .groupby("坪數區間", observed=True)["單價(萬/坪)"]
        .agg(中位數="median", 樣本數="count")
        .reindex(size_labels).dropna().reset_index()
    )
    agg_b["標註"] = agg_b.apply(
        lambda r: f"{r['中位數']:.1f}<br>n={int(r['樣本數']):,}", axis=1,
    )

    if len(agg_b) >= 3:
        small_v = agg_b.iloc[0]["中位數"]
        large_v = agg_b.iloc[-1]["中位數"]
        middle_min_v = agg_b.iloc[1:-1]["中位數"].min()
        cheapest_row = agg_b.loc[agg_b["中位數"].idxmin()]
        if small_v > middle_min_v and large_v > middle_min_v:
            st.info(
                f"單坪價格呈現 U 型："
                f"**{agg_b.iloc[0]['坪數區間']}** 小套房（{small_v:.1f} 萬）和 "
                f"**{agg_b.iloc[-1]['坪數區間']}** 豪宅（{large_v:.1f} 萬）較貴，"
                f"**{cheapest_row['坪數區間']}** 最便宜（{cheapest_row['中位數']:.1f} 萬）。"
            )
        else:
            most_exp = agg_b.loc[agg_b["中位數"].idxmax()]
            st.info(
                f"**{most_exp['坪數區間']}** 單價最高（{most_exp['中位數']:.1f} 萬/坪），"
                f"**{cheapest_row['坪數區間']}** 最低（{cheapest_row['中位數']:.1f} 萬/坪）。"
            )
    elif len(agg_b) >= 2:
        most_exp = agg_b.loc[agg_b["中位數"].idxmax()]
        cheapest_row = agg_b.loc[agg_b["中位數"].idxmin()]
        st.info(
            f"**{most_exp['坪數區間']}** 單價較高（{most_exp['中位數']:.1f} 萬/坪），"
            f"**{cheapest_row['坪數區間']}** 較低（{cheapest_row['中位數']:.1f} 萬/坪）。"
        )

    fig_size_bar = px.bar(
        agg_b, x="坪數區間", y="中位數",
        color="中位數", color_continuous_scale=CHART_PALETTE,
        text="標註",
        category_orders={"坪數區間": size_labels},
    )
    fig_size_bar.update_traces(
        textposition="outside",
        cliponaxis=False,
        customdata=[[c] for c in agg_b["樣本數"].tolist()],
        hovertemplate=(
            "坪數區間：%{x}<br>"
            "中位數單價：%{y:,.1f} 萬/坪<br>"
            "樣本數：%{customdata[0]:,} 筆"
            "<extra></extra>"
        ),
    )
    fig_size_bar.update_layout(
        coloraxis_showscale=False,
        yaxis_title="中位數單價（萬/坪）",
        xaxis_title=None,
    )
    style_chart(fig_size_bar, height=420)
    st.plotly_chart(fig_size_bar, use_container_width=True)

    st.caption("以下為篩選後資料的隨機 100 筆樣本，可向右滑動檢視所有欄位")
    with st.expander("看原始樣本（隨機 100 筆）"):
        n_show = min(100, len(sub))
        st.dataframe(
            sub.sample(n_show, random_state=42).reset_index(drop=True),
            use_container_width=True,
        )


def page_predict(meta: dict, pipe) -> None:
    """房價試算頁面。"""
    st.markdown("## 房價試算")

    metrics = meta["selected_metrics"]
    acc_pct = metrics["r2"] * 100
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "模型準確度", f"{acc_pct:.0f}%",
        help="模型在測試資料上能解釋的房價變動比例。100% 為完美預測，0% 為毫無預測能力。",
    )
    m2.metric(
        "平均誤差", f"±{metrics['mae']:.1f} 萬/坪",
        help="預測值與實際成交價的平均差距（MAE）",
    )
    m3.metric(
        "最大典型誤差", f"±{metrics['rmse']:.1f} 萬/坪",
        help="較大誤差的代表值，受極端案例影響較高（RMSE）",
    )
    st.info(
        f"""**模型表現解讀**
- 本模型可解釋約 **{acc_pct:.0f}% 的房價變動**，剩餘 {100 - acc_pct:.0f}% 來自地段細節、屋況、裝潢、捷運距離等未納入的因素
- 對比基準：簡單線性迴歸約 40%；商用估價系統約 85%+
- 下一階段可加入：行政區內街廓、捷運距離、學區等地段細粒度特徵"""
    )

    st.markdown("### 試算條件")
    with st.form("predict_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            district = st.selectbox("行政區", meta["districts"])
            building_type = st.selectbox("建物型態", meta["building_types"])
            ping = st.number_input(
                "總坪數（權狀，含公設）",
                min_value=5.0, max_value=200.0, value=30.0, step=1.0,
                help=(
                    "資料來源為內政部實價登錄之建物移轉總面積，"
                    "為權狀坪數（包含主建物+附屬建物+公設）。"
                    "台灣住宅大樓公設比常見 30~35%。"
                ),
            )
        with col_b:
            age = st.number_input("屋齡（年）", min_value=0, max_value=80, value=15, step=1)
            floor = st.number_input("樓層", min_value=-5, max_value=80, value=5, step=1)
        submitted = st.form_submit_button("試算", type="primary")

    if submitted:
        X = pd.DataFrame([{
            "行政區": district,
            "建物型態": building_type,
            "總坪數": float(ping),
            "屋齡": float(age),
            "樓層": float(floor),
        }])
        unit_price = float(pipe.predict(X)[0])
        total_price_wan = unit_price * ping
        total_price_yi = total_price_wan / 10000

        r1, r2 = st.columns(2)
        r1.metric("預測單價（萬/坪，權狀）", f"{unit_price:,.2f}")
        if total_price_yi >= 1:
            r2.metric("預測總價", f"{total_price_yi:,.2f} 億元")
        else:
            r2.metric("預測總價", f"{total_price_wan:,.0f} 萬元")

        lo = unit_price - metrics["rmse"]
        hi = unit_price + metrics["rmse"]
        st.caption(f"參考誤差區間（±RMSE）：{lo:,.1f} ~ {hi:,.1f} 萬/坪")

    st.divider()
    st.caption(
        f"模型：{meta['best_model']}　·　訓練樣本 {meta['n_train']:,}　·　"
        f"測試樣本 {meta['n_test']:,}"
    )


def main() -> None:
    inject_styles()
    page = render_sidebar()
    df = load_data()
    render_topbar(df)
    if page == "資料分布概況":
        page_explore(df)
    else:
        pipe, meta = load_model_resources()
        page_predict(meta, pipe)


if __name__ == "__main__":
    main()
