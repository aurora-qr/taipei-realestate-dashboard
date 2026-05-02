"""台北市實價登錄分析儀表板 (Streamlit)。

兩頁：
1. 資料探索：行政區篩選、KPI、單價分布、坪數 vs 單價散布圖
2. 房價預測：依使用者輸入特徵，預測單價(萬/坪)與總價
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from src import model as model_mod
from src import data_pipeline as dp

st.set_page_config(
    page_title="台北市實價登錄分析儀表板",
    page_icon="🏙️",
    layout="wide",
)

DB_PATH = dp.DEFAULT_DB_PATH


@st.cache_data(show_spinner="載入資料中…")
def load_data() -> pd.DataFrame:
    """從 SQLite 讀取清整資料；若資料庫不存在則先跑一次 pipeline。"""
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


def render_sidebar() -> str:
    """側邊欄頁面導覽。"""
    st.sidebar.title("🏙️ 台北實價登錄")
    st.sidebar.caption("Taipei Real Estate Dashboard")
    page = st.sidebar.radio("選擇頁面", ["資料探索", "房價預測"])
    return page


def render_footer(df: pd.DataFrame) -> None:
    """頁面底部一行小字資料來源說明。"""
    st.divider()
    st.caption(
        f"資料來源：內政部不動產交易實價查詢服務網 · "
        f"台北市 2025Q2–2026Q1 · 共 {len(df):,} 筆有效樣本"
    )


def page_explore(df: pd.DataFrame) -> None:
    """資料探索頁面。"""
    st.title("📊 資料探索")
    st.caption("依行政區篩選台北市住宅交易，觀察單價分布與規模關係。")

    districts = sorted(df["行政區"].unique().tolist())
    selected = st.multiselect(
        "行政區（可多選，預設全選）",
        options=districts,
        default=districts,
    )

    if not selected:
        st.warning("請至少選一個行政區。")
        return

    sub = df[df["行政區"].isin(selected)]

    c1, c2, c3 = st.columns(3)
    c1.metric("平均單價 (萬/坪)", f"{sub['單價(萬/坪)'].mean():.2f}")
    c2.metric("中位數 (萬/坪)", f"{sub['單價(萬/坪)'].median():.2f}")
    c3.metric("樣本數", f"{len(sub):,}")

    st.subheader("單價分布")
    median_price = float(sub["單價(萬/坪)"].median())
    fig_hist = px.histogram(
        sub, x="單價(萬/坪)", nbins=40,
        color_discrete_sequence=["#4C78A8"],
        title=None,
    )
    # 中位數虛線 + 標註
    fig_hist.add_vline(
        x=median_price,
        line_dash="dash",
        line_color="#D62728",
        line_width=2,
        annotation_text=f"中位數 {median_price:.1f}",
        annotation_position="top right",
        annotation_font_color="#D62728",
    )
    fig_hist.update_layout(
        bargap=0.05, height=380,
        yaxis_title="交易筆數",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("各行政區單價排名（中位數）")
    # 計算每區中位數與樣本數，依中位數降冪
    agg_d = (
        sub.groupby("行政區")["單價(萬/坪)"]
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
            f"💡 {high_d}單價最高（約 {high_v:.1f} 萬/坪），"
            f"{low_d}最低（約 {low_v:.1f} 萬/坪），"
            f"全市最大差距約 {high_v - low_v:.1f} 萬/坪。"
        )
    else:
        only = agg_d.iloc[0]
        st.info(f"💡 {only['行政區']}中位數單價約 {only['中位數']:.1f} 萬/坪。")

    fig_dist = px.bar(
        agg_d, x="行政區", y="中位數",
        color="中位數", color_continuous_scale="Blues",
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
        height=440,
        coloraxis_showscale=False,
        yaxis_title="中位數單價 (萬/坪)",
        xaxis_title=None,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("不同坪數的單價比較")
    st.caption("※ 坪數均為權狀坪數（含公設），與內政部實價登錄、各大房仲網站口徑一致")
    size_bins = [5, 15, 30, 50, 100, float("inf")]
    size_labels = ["5-15坪", "15-30坪", "30-50坪", "50-100坪", "100坪以上"]
    sub_with_bin = sub.copy()
    sub_with_bin["坪數區間"] = pd.cut(
        sub_with_bin["總坪數"],
        bins=size_bins, labels=size_labels,
        right=False, include_lowest=True,
    )
    agg_b = (
        sub_with_bin.dropna(subset=["坪數區間"])
        .groupby("坪數區間", observed=True)["單價(萬/坪)"]
        .agg(中位數="median", 樣本數="count")
        .reindex(size_labels)
        .dropna()
        .reset_index()
    )
    agg_b["標註"] = agg_b.apply(
        lambda r: f"{r['中位數']:.1f}<br>n={int(r['樣本數']):,}", axis=1,
    )

    # 動態洞察：若呈現 U 型則用 U 型敘述，否則回報極值
    if len(agg_b) >= 3:
        small_v = agg_b.iloc[0]["中位數"]
        large_v = agg_b.iloc[-1]["中位數"]
        middle_min_v = agg_b.iloc[1:-1]["中位數"].min()
        cheapest_row = agg_b.loc[agg_b["中位數"].idxmin()]
        if small_v > middle_min_v and large_v > middle_min_v:
            st.info(
                f"💡 單坪價格呈現 U 型："
                f"{agg_b.iloc[0]['坪數區間']} 小套房（{small_v:.1f} 萬）和 "
                f"{agg_b.iloc[-1]['坪數區間']} 豪宅（{large_v:.1f} 萬）較貴，"
                f"{cheapest_row['坪數區間']} 主力坪數最便宜（{cheapest_row['中位數']:.1f} 萬）。"
            )
        else:
            most_exp = agg_b.loc[agg_b["中位數"].idxmax()]
            st.info(
                f"💡 不同坪數中，{most_exp['坪數區間']} 單價最高（{most_exp['中位數']:.1f} 萬/坪），"
                f"{cheapest_row['坪數區間']} 最低（{cheapest_row['中位數']:.1f} 萬/坪）。"
            )
    elif len(agg_b) >= 2:
        most_exp = agg_b.loc[agg_b["中位數"].idxmax()]
        cheapest_row = agg_b.loc[agg_b["中位數"].idxmin()]
        st.info(
            f"💡 {most_exp['坪數區間']} 單價較高（{most_exp['中位數']:.1f} 萬/坪），"
            f"{cheapest_row['坪數區間']} 較低（{cheapest_row['中位數']:.1f} 萬/坪）。"
        )

    fig_size_bar = px.bar(
        agg_b, x="坪數區間", y="中位數",
        color="中位數", color_continuous_scale="Blues",
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
        height=420,
        coloraxis_showscale=False,
        yaxis_title="中位數單價 (萬/坪)",
        xaxis_title=None,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_size_bar, use_container_width=True)

    st.caption("以下為篩選後資料的隨機 100 筆樣本，可向右滑動檢視所有欄位")
    with st.expander("看原始樣本（隨機 100 筆）"):
        n_show = min(100, len(sub))
        st.dataframe(
            sub.sample(n_show, random_state=42).reset_index(drop=True),
            use_container_width=True,
        )


def page_predict(meta: dict, pipe) -> None:
    """房價預測頁面。"""
    st.title("🔮 房價預測")
    st.caption(f"使用模型：**{meta['best_model']}**　| 訓練樣本 {meta['n_train']:,}　測試樣本 {meta['n_test']:,}")

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
        f"""💡 **模型表現解讀**
- 本模型可解釋約 **{acc_pct:.0f}% 的房價變動**，剩餘 {100 - acc_pct:.0f}% 來自地段細節、屋況、裝潢、捷運距離等未納入的因素
- 對比基準：簡單線性迴歸約 40%；商用估價系統約 85%+
- 下一階段可加入：行政區內街廓、捷運距離、學區等地段細粒度特徵"""
    )

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
        submitted = st.form_submit_button("預測", type="primary")

    if submitted:
        X = pd.DataFrame([{
            "行政區": district,
            "建物型態": building_type,
            "總坪數": float(ping),
            "屋齡": float(age),
            "樓層": float(floor),
        }])
        unit_price = float(pipe.predict(X)[0])
        total_price_wan = unit_price * ping  # 萬元
        total_price_yi = total_price_wan / 10000  # 億

        r1, r2 = st.columns(2)
        r1.metric("預測單價 (萬/坪，權狀)", f"{unit_price:,.2f}")
        if total_price_yi >= 1:
            r2.metric("預測總價", f"{total_price_yi:,.2f} 億元")
        else:
            r2.metric("預測總價", f"{total_price_wan:,.0f} 萬元")

        lo = unit_price - metrics["rmse"]
        hi = unit_price + metrics["rmse"]
        st.caption(f"參考誤差區間（±RMSE）：{lo:,.1f} ~ {hi:,.1f} 萬/坪")


def main() -> None:
    page = render_sidebar()
    df = load_data()
    if page == "資料探索":
        page_explore(df)
    else:
        pipe, meta = load_model_resources()
        page_predict(meta, pipe)
    render_footer(df)


if __name__ == "__main__":
    main()
