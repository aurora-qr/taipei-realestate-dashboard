# 台北市不動產儀表板

以 Streamlit 打造的互動式儀表板，用來探索與預測台北市住宅實價登錄行情。
資料來源為內政部不動產成交案件實際資訊資料集（MOI），專案涵蓋從原始
CSV 匯入、資料清洗、模型訓練到互動式前端的完整資料管線。

## 線上 Demo

https://aurora-realestate.streamlit.app

## 功能特色

- 針對 MOI 實價登錄 CSV 的清洗管線（處理民國年轉換、中文地址解析、離群值過濾）
- 以 SQLite 為後端的查詢層
- 使用 Random Forest 進行房價預測（測試集 R² ≈ 0.56）
- 雙頁面 Streamlit UI：資料探索與房價估算

## 技術堆疊

Python 3.11、pandas、scikit-learn、Streamlit、Plotly、SQLite、joblib。

## 專案結構

```
src/
  data_pipeline.py    # ETL: raw CSV -> SQLite
  model.py            # train and persist regression models
app.py                # Streamlit entry point
requirements.txt
data.db               # generated, gitignored
models/               # generated, gitignored
raw_*.csv             # MOI source files, gitignored
```

## 資料說明

資料來源：內政部不動產成交案件實際資訊資料集
(https://plvr.land.moi.gov.tw/DownloadOpenData)

涵蓋範圍：台北市住宅交易，2025 Q2 至 2026 Q1，原始約 22,000 筆紀錄，
清洗後保留約 9,500 筆。

清洗步驟僅保留住家用、住商用及見其他登記事項等用途類別，將民國年轉為西元年，
從地址字串中擷取行政區，將平方公尺換算為坪，並移除離群值（單價落在第 5–95
百分位之外、面積不在 5–200 坪區間、屋齡不在 0–80 年區間者）。

## 本機執行

```bash
git clone https://github.com/aurora-qr/taipei-realestate-dashboard.git
cd taipei-realestate-dashboard
pip install -r requirements.txt
python src/data_pipeline.py    # build data.db
python src/model.py            # train models
streamlit run app.py
```

執行資料管線之前，需要先將 MOI 季度 CSV 檔放到專案根目錄，
命名為 `raw_2025Q2.csv`、`raw_2025Q3.csv` 等。

## 備註

價格單位為萬元/坪（含公設的主建物面積），與 MOI 及台灣主要房屋交易平台
的慣用單位一致。
